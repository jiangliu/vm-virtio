// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE-BSD-3-Clause file.
//
// Copyright © 2019 Intel Corporation
//
// Copyright (C) 2020 Alibaba Cloud. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

use std::cmp::min;
use std::fmt::{self, Display};
use std::mem::size_of;
use std::num::Wrapping;
use std::ops::Deref;
use std::sync::atomic::{fence, AtomicU16, AtomicU32, Ordering};

use vm_memory::{Address, ByteValued, GuestAddress, GuestMemory, GuestUsize, VolatileMemory};

pub(super) const VIRTQ_DESC_F_NEXT: u16 = 0x1;
pub(super) const VIRTQ_DESC_F_WRITE: u16 = 0x2;
pub(super) const VIRTQ_DESC_F_INDIRECT: u16 = 0x4;

const VIRTQ_USED_ELEMENT_SIZE: usize = 8;
// Used ring header: flags (u16) + idx (u16)
const VIRTQ_USED_RING_HEADER_SIZE: usize = 4;
// This is the size of the used ring metadata: header + used_event (u16).
// The total size of the used ring is:
// VIRTQ_USED_RING_HMETA_SIZE + VIRTQ_USED_ELEMENT_SIZE * queue_size
const VIRTQ_USED_RING_META_SIZE: usize = VIRTQ_USED_RING_HEADER_SIZE + 2;
// Used flags
const VIRTQ_USED_F_NO_NOTIFY: u16 = 0x1;

const VIRTQ_AVAIL_ELEMENT_SIZE: usize = 2;
// Avail ring header: flags(u16) + idx(u16)
const VIRTQ_AVAIL_RING_HEADER_SIZE: usize = 4;
// This is the size of the available ring metadata: header + avail_event (u16).
// The total size of the available ring is:
// VIRTQ_AVAIL_RING_META_SIZE + VIRTQ_AVAIL_ELEMENT_SIZE * queue_size
const VIRTQ_AVAIL_RING_META_SIZE: usize = VIRTQ_AVAIL_RING_HEADER_SIZE + 2;

// AtomicU16::load() will be used to fetch the descriptor, and AtomicU16 is
// returned by VolatileSlice::get_atomic_ref(), which has an explicit constraint
// that the entire descriptor doesn't cross the page boundary. Otherwise the
// descriptor may be split into two mmap regions which causes failure of
// VolatileSlice::get_atomic_ref().
//
// The Virtio Spec 1.0 defines the alignment of VirtIO descriptor is 16 bytes,
// which fulfills the explicit constraint of VolatileSlice::get_atomic_ref().
const VIRTQ_DESCRIPTOR_SIZE: usize = 16;

/// Virtio Queue related errors.
#[derive(Debug)]
pub enum Error {
    /// Failed to access guest memory.
    GuestMemoryError,
    /// Invalid indirect descriptor.
    InvalidIndirectDescriptor,
    /// Invalid descriptor chain.
    InvalidChain,
}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::Error::*;

        match self {
            GuestMemoryError => write!(f, "error accessing guest memory"),
            InvalidChain => write!(f, "invalid descriptor chain"),
            InvalidIndirectDescriptor => write!(f, "invalid indirect descriptor"),
        }
    }
}

impl std::error::Error for Error {}

/// A virtio descriptor constraints with C representation
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct Descriptor {
    /// Guest physical address of device specific data
    addr: u64,

    /// Length of device specific data
    len: u32,

    /// Includes next, write, and indirect bits
    flags: u16,

    /// Index into the descriptor table of the next descriptor if flags has
    /// the next bit set
    next: u16,
}

#[allow(clippy::len_without_is_empty)]
impl Descriptor {
    /// Return the guest physical address of descriptor buffer
    pub fn addr(&self) -> GuestAddress {
        GuestAddress(self.addr)
    }

    /// Return the length of descriptor buffer
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Return the flags for this descriptor, including next, write and indirect
    /// bits
    pub fn flags(&self) -> u16 {
        self.flags
    }

    /// Checks if the driver designated this as a write only descriptor.
    ///
    /// If this is false, this descriptor is read only.
    /// Write only means the the emulated device can write and the driver can read.
    pub fn is_write_only(&self) -> bool {
        self.flags & VIRTQ_DESC_F_WRITE != 0
    }

    /// Checks if this descriptor has another descriptor linked after it.
    pub fn has_next(&self) -> bool {
        self.flags & VIRTQ_DESC_F_NEXT != 0
    }
}

unsafe impl ByteValued for Descriptor {}

/// A virtio descriptor chain.
pub struct DescriptorChain<'a, M: GuestMemory> {
    mem: &'a M,
    desc_table: GuestAddress,
    queue_size: u16,
    ttl: u16,   // used to prevent infinite chain cycles
    index: u16, // descriptor index of the chain header

    /// The current descriptor
    desc: Descriptor,
    curr_indirect: Option<Box<DescriptorChain<'a, M>>>,
    is_master: bool,
    has_next: bool,
}

impl<'a, M: GuestMemory> DescriptorChain<'a, M> {
    fn read_new(
        mem: &'a M,
        desc_table: GuestAddress,
        queue_size: u16,
        ttl: u16,
        index: u16,
    ) -> Option<Self> {
        if index >= queue_size {
            return None;
        }

        let desc_addr = match desc_table.checked_add(VIRTQ_DESCRIPTOR_SIZE as u64 * index as u64) {
            Some(a) => a,
            None => return None,
        };
        // The descriptor is 16 bytes and aligned on on 16-bytes, so it won't cross guest memory boundary.
        let slice = mem.get_slice(desc_addr, VIRTQ_DESCRIPTOR_SIZE).ok()?;
        let desc = slice.get_ref(0).ok()?.load();
        let chain = DescriptorChain {
            mem,
            desc_table,
            queue_size,
            ttl,
            index,
            desc,
            curr_indirect: None,
            is_master: true,
            has_next: true,
        };

        if chain.is_valid() {
            Some(chain)
        } else {
            None
        }
    }

    /// Create a new DescriptorChain instance.
    pub fn checked_new(
        mem: &'a M,
        dtable_addr: GuestAddress,
        queue_size: u16,
        index: u16,
    ) -> Option<Self> {
        Self::read_new(mem, dtable_addr, queue_size, queue_size, index)
    }

    /// Create a `DescriptorChain` from the indirect target descriptor table.
    pub fn new_from_indirect(&self) -> Result<DescriptorChain<'a, M>, Error> {
        if !self.is_indirect() {
            return Err(Error::InvalidIndirectDescriptor);
        }

        let desc_head = self.desc.addr;
        let desc_len = self.desc.len as usize;
        // Check the target indirect descriptor table is correctly aligned.
        if desc_head & (VIRTQ_DESCRIPTOR_SIZE as u64 - 1) != 0
            || desc_len & (VIRTQ_DESCRIPTOR_SIZE - 1) != 0
            || desc_len < VIRTQ_DESCRIPTOR_SIZE
            || desc_len / VIRTQ_DESCRIPTOR_SIZE > std::u16::MAX as usize
        {
            return Err(Error::InvalidIndirectDescriptor);
        }

        // These reads can't fail unless Guest memory is hopelessly broken.
        let desc: Descriptor = self
            .mem
            .get_slice(GuestAddress(desc_head), VIRTQ_DESCRIPTOR_SIZE)
            .map(|s| s.get_ref(0).unwrap().load())
            .map_err(|_| Error::GuestMemoryError)?;

        let chain = DescriptorChain {
            mem: self.mem,
            desc_table: GuestAddress(desc_head),
            queue_size: (desc_len / VIRTQ_DESCRIPTOR_SIZE) as u16,
            ttl: (desc_len / VIRTQ_DESCRIPTOR_SIZE) as u16,
            index: self.index,
            desc: Descriptor {
                addr: desc.addr,
                len: desc.len,
                flags: desc.flags,
                next: desc.next,
            },
            curr_indirect: None,
            is_master: false,
            has_next: true,
        };

        if !chain.is_valid() {
            return Err(Error::InvalidChain);
        }

        Ok(chain)
    }

    fn is_valid(&self) -> bool {
        self.mem
            .checked_offset(self.desc.addr(), self.desc.len as usize)
            .filter(|_| !self.desc.has_next() || self.desc.next < self.queue_size)
            .is_some()
    }

    /// Get the descriptor index of the chain header
    pub fn index(&self) -> u16 {
        self.index
    }

    /// Checks if this descriptor chain has another descriptor chain linked after it.
    pub fn has_next(&self) -> bool {
        self.has_next || self.curr_indirect.is_some()
    }

    /// Checks if the descriptor is an indirect descriptor.
    pub fn is_indirect(&self) -> bool {
        self.desc.flags & VIRTQ_DESC_F_INDIRECT != 0
    }

    /// Return a `GuestMemory` object that can be used to access the buffers
    /// pointed to by the descriptor chain.
    pub fn memory(&self) -> &M {
        &*self.mem
    }

    /// Returns an iterator that only yields the readable descriptors in the chain.
    pub fn readable(self) -> DescriptorChainRwIter<'a, M> {
        DescriptorChainRwIter {
            chain: self,
            writable: false,
        }
    }

    /// Returns an iterator that only yields the writable descriptors in the chain.
    pub fn writable(self) -> DescriptorChainRwIter<'a, M> {
        DescriptorChainRwIter {
            chain: self,
            writable: true,
        }
    }
}

impl<'a, M: GuestMemory> Iterator for DescriptorChain<'a, M> {
    type Item = Descriptor;

    /// Returns the next descriptor in this descriptor chain, if there is one.
    ///
    /// Note that this is distinct from the next descriptor chain returned by
    /// [`AvailIter`](struct.AvailIter.html), which is the head of the next
    /// _available_ descriptor chain.
    fn next(&mut self) -> Option<Self::Item> {
        if self.ttl == 0 {
            return None;
        }

        // Special handling for indirect descriptor table
        if self.is_indirect() {
            // An indirect descriptor can not refer to another indirect descriptor table
            if !self.is_master {
                return None;
            }
            if self.curr_indirect.is_none() {
                let indirect_chain = self.new_from_indirect().ok()?;
                self.curr_indirect = Some(Box::new(indirect_chain));
            }
            // Above code ensures that it's safe to unwrap().
            let indirect = self.curr_indirect.as_mut().unwrap();

            match indirect.next() {
                // return the next descriptor from the indirect chain
                Some(d) => Some(d),
                // current indirect chain hs reached the end, return to the master chain.
                None => {
                    self.curr_indirect = None;
                    if !self.has_next() {
                        // the main descriptor has reached the end too.
                        self.ttl = 0;
                        None
                    } else {
                        // read next descriptor from the main descriptor chain.
                        let index = self.desc.next;
                        let offset = index as u64 * VIRTQ_DESCRIPTOR_SIZE as u64;
                        let addr = self.desc_table.unchecked_add(offset);
                        let slice = self.mem.get_slice(addr, VIRTQ_DESCRIPTOR_SIZE).ok()?;

                        self.desc = slice.get_ref(0).ok()?.load();
                        self.ttl -= 1;
                        self.next()
                    }
                }
            }
        } else {
            let curr = self.desc;
            if !curr.has_next() {
                self.ttl = 0
            } else {
                let index = self.desc.next;
                let desc_table_size = VIRTQ_DESCRIPTOR_SIZE * self.queue_size as usize;
                let slice = self.mem.get_slice(self.desc_table, desc_table_size).ok()?;
                self.desc = slice
                    .get_array_ref(0, self.queue_size as usize)
                    .ok()?
                    .load(index as usize);
                self.ttl -= 1;
            }

            self.has_next = curr.has_next();

            Some(curr)
        }
    }
}

impl<'a, M: GuestMemory> Clone for DescriptorChain<'a, M> {
    fn clone(&self) -> Self {
        DescriptorChain {
            mem: self.mem,
            desc_table: self.desc_table,
            queue_size: self.queue_size,
            ttl: self.ttl,
            index: self.index,
            desc: self.desc,
            curr_indirect: self.curr_indirect.clone(),
            is_master: self.is_master,
            has_next: self.has_next,
        }
    }
}

/// An iterator for readable or writable descriptors.
pub struct DescriptorChainRwIter<'a, M: GuestMemory> {
    chain: DescriptorChain<'a, M>,
    writable: bool,
}

impl<'a, M: GuestMemory> Iterator for DescriptorChainRwIter<'a, M> {
    type Item = Descriptor;

    /// Returns the next descriptor in this descriptor chain, if there is one.
    ///
    /// Note that this is distinct from the next descriptor chain returned by
    /// [`AvailIter`](struct.AvailIter.html), which is the head of the next
    /// _available_ descriptor chain.
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.chain.next() {
                Some(v) => {
                    if v.is_write_only() == self.writable {
                        return Some(v);
                    }
                }
                None => return None,
            }
        }
    }
}

/// Consuming iterator over all available descriptor chain heads in the queue.
pub struct AvailIter<'a, 'b, M: GuestMemory> {
    mem: &'a M,
    desc_table: GuestAddress,
    avail_ring: GuestAddress,
    next_index: Wrapping<u16>,
    last_index: Wrapping<u16>,
    queue_size: u16,
    next_avail: &'b mut Wrapping<u16>,
}

impl<'a, 'b, M: GuestMemory> AvailIter<'a, 'b, M> {
    /// Constructs an empty descriptor iterator.
    pub fn new(mem: &'a M, q_next_avail: &'b mut Wrapping<u16>) -> AvailIter<'a, 'b, M> {
        AvailIter {
            mem,
            desc_table: GuestAddress(0),
            avail_ring: GuestAddress(0),
            next_index: Wrapping(0),
            last_index: Wrapping(0),
            queue_size: 0,
            next_avail: q_next_avail,
        }
    }
}

impl<'a, 'b, M: GuestMemory> Iterator for AvailIter<'a, 'b, M> {
    type Item = DescriptorChain<'a, M>;

    fn next(&mut self) -> Option<Self::Item> {
        let next_index = self.next_index.0 % self.queue_size;
        if next_index == self.last_index.0 % self.queue_size {
            return None;
        }

        let offset = (VIRTQ_AVAIL_RING_HEADER_SIZE as u16
            + (self.next_index.0 % self.queue_size) * VIRTQ_AVAIL_ELEMENT_SIZE as u16)
            as usize;
        let avail_addr = self.avail_ring.checked_add(offset as u64)?;
        // This index is checked below in checked_new
        let desc_index: u16 = match vq_load_u16(self.mem.deref(), avail_addr) {
            Ok(index) => index,
            Err(e) => {
                error!(
                    "Failed to read desc_index from avail_addr ({:?}): {:?}",
                    avail_addr.raw_value(),
                    e
                );
                return None;
            }
        };

        self.next_index += Wrapping(1);

        let desc =
            DescriptorChain::checked_new(self.mem, self.desc_table, self.queue_size, desc_index);
        if desc.is_some() {
            *self.next_avail += Wrapping(1);
        } else {
            error!("Received invalid descriptor, no way to recover!");
        }
        desc
    }
}

#[derive(Clone)]
/// A virtio queue's parameters.
pub struct Queue {
    /// The maximal size in elements offered by the device
    max_size: u16,

    /// The next available index
    pub next_avail: Wrapping<u16>,

    /// The next used index
    pub next_used: Wrapping<u16>,

    /// Notification from driver is enabled.
    event_notification_enabled: bool,

    /// VIRTIO_F_RING_EVENT_IDX negotiated
    event_idx_enabled: bool,

    /// The last used value when using EVENT_IDX
    signalled_used: Option<Wrapping<u16>>,

    /// The queue size in elements the driver selected
    pub size: u16,

    /// Indicates if the queue is finished with configuration
    pub ready: bool,

    /// Guest physical address of the descriptor table
    pub desc_table: GuestAddress,

    /// Guest physical address of the available ring
    pub avail_ring: GuestAddress,

    /// Guest physical address of the used ring
    pub used_ring: GuestAddress,
}

impl Queue {
    /// Constructs an empty virtio queue with the given `max_size`.
    pub fn new(max_size: u16) -> Queue {
        Queue {
            max_size,
            size: max_size,
            ready: false,
            desc_table: GuestAddress(0),
            avail_ring: GuestAddress(0),
            used_ring: GuestAddress(0),
            next_avail: Wrapping(0),
            next_used: Wrapping(0),
            event_notification_enabled: true,
            event_idx_enabled: false,
            signalled_used: None,
        }
    }

    /// Gets the virtio queue maximum size.
    pub fn max_size(&self) -> u16 {
        self.max_size
    }

    /// Return the actual size of the queue, as the driver may not set up a
    /// queue as big as the device allows.
    pub fn actual_size(&self) -> u16 {
        min(self.size, self.max_size)
    }

    /// Reset the queue to a state that is acceptable for a device reset
    pub fn reset(&mut self) {
        self.ready = false;
        self.size = self.max_size;
    }

    /// Enable/disable the VIRTIO_F_RING_EVENT_IDX feature.
    pub fn set_event_idx(&mut self, enabled: bool) {
        /* Also reset the last signalled event */
        self.signalled_used = None;
        self.event_idx_enabled = enabled;
    }

    /// Check if the virtio queue configuration is valid.
    pub fn is_valid<M: GuestMemory>(&self, mem: &M) -> bool {
        let queue_size = self.actual_size() as usize;
        let desc_table = self.desc_table;
        let desc_table_size = VIRTQ_DESCRIPTOR_SIZE * queue_size;
        let avail_ring = self.avail_ring;
        let avail_ring_size = VIRTQ_AVAIL_RING_META_SIZE + VIRTQ_AVAIL_ELEMENT_SIZE * queue_size;
        let used_ring = self.used_ring;
        let used_ring_size = VIRTQ_USED_RING_META_SIZE + VIRTQ_USED_ELEMENT_SIZE * queue_size;
        if !self.ready {
            error!("attempt to use virtio queue that is not marked ready");
            false
        } else if self.size > self.max_size || self.size == 0 || (self.size & (self.size - 1)) != 0
        {
            error!("virtio queue with invalid size: {}", self.size);
            false
        } else if desc_table
            .checked_add(desc_table_size as GuestUsize)
            .map_or(true, |v| !mem.address_in_range(v))
        {
            error!(
                "virtio queue descriptor table goes out of bounds: start:0x{:08x} size:0x{:08x}",
                desc_table.raw_value(),
                desc_table_size
            );
            false
        } else if avail_ring
            .checked_add(avail_ring_size as GuestUsize)
            .map_or(true, |v| !mem.address_in_range(v))
        {
            error!(
                "virtio queue available ring goes out of bounds: start:0x{:08x} size:0x{:08x}",
                avail_ring.raw_value(),
                avail_ring_size
            );
            false
        } else if used_ring
            .checked_add(used_ring_size as GuestUsize)
            .map_or(true, |v| !mem.address_in_range(v))
        {
            error!(
                "virtio queue used ring goes out of bounds: start:0x{:08x} size:0x{:08x}",
                used_ring.raw_value(),
                used_ring_size
            );
            false
        } else if desc_table.mask(0xf) != 0 {
            error!("virtio queue descriptor table breaks alignment contraints");
            false
        } else if avail_ring.mask(0x1) != 0 {
            error!("virtio queue available ring breaks alignment contraints");
            false
        } else if used_ring.mask(0x3) != 0 {
            error!("virtio queue used ring breaks alignment contraints");
            false
        } else {
            true
        }
    }

    /// A consuming iterator over all available descriptor chain heads offered by the driver.
    pub fn iter<'a, 'b, M: GuestMemory>(&'b mut self, mem: &'a M) -> AvailIter<'a, 'b, M> {
        let queue_size = self.actual_size();
        let avail_ring = self.avail_ring;

        let index_addr = match avail_ring.checked_add(2) {
            Some(ret) => ret,
            None => {
                warn!("Invalid offset {}", avail_ring.raw_value());
                return AvailIter::new(mem, &mut self.next_avail);
            }
        };
        // Note that last_index has no invalid values
        let last_index: u16 = match vq_load_u16(mem.deref(), index_addr) {
            Ok(index) => index,
            Err(_) => return AvailIter::new(mem, &mut self.next_avail),
        };

        AvailIter {
            mem,
            desc_table: self.desc_table,
            avail_ring,
            next_index: self.next_avail,
            last_index: Wrapping(last_index),
            queue_size,
            next_avail: &mut self.next_avail,
        }
    }

    /// Puts an available descriptor head into the used ring for use by the guest.
    pub fn add_used<M: GuestMemory>(&mut self, mem: &M, desc_index: u16, len: u32) -> Option<u16> {
        if desc_index >= self.actual_size() {
            error!(
                "attempted to add out of bounds descriptor to used ring: {}",
                desc_index
            );
            return None;
        }

        let used_ring = self.used_ring;
        let next_used = u64::from(self.next_used.0 % self.actual_size());
        let used_elem = used_ring.unchecked_add(
            VIRTQ_USED_RING_HEADER_SIZE as u64 + next_used * VIRTQ_USED_ELEMENT_SIZE as u64,
        );

        // These writes can't fail as we are guaranteed to be within the descriptor ring.
        vq_store_u32(mem.deref(), u32::from(desc_index), used_elem)
            .expect("update used_elem desc_index");
        vq_store_u32(mem.deref(), len as u32, used_elem.unchecked_add(4))
            .expect("update used_elem len");

        self.next_used += Wrapping(1);

        // This fence ensures all descriptor writes are visible before the index update is.
        fence(Ordering::Release);

        // We are guaranteed to be within the used ring, this write can't fail.
        vq_store_u16(mem.deref(), self.next_used.0, used_ring.unchecked_add(2))
            .expect("update used_ring next_used");

        Some(self.next_used.0)
    }

    /// Update avail_event on the used ring with the last index in the avail ring.
    ///
    /// The device can suppress notifications in a manner analogous to the way drivers can suppress
    /// interrupts. The device manipulates flags or avail_event in the used ring the same way the
    /// driver manipulates flags or used_event in the available ring.
    ///
    /// The device MAY use avail_event to advise the driver that notifications are unnecessary until
    /// the driver writes entry with an index specified by avail_event into the available ring
    /// (equivalently, until idx in the available ring will reach the value avail_event + 1).
    fn update_avail_event<M: GuestMemory>(&mut self, mem: &M) {
        // Safe because we have validated the queue and access guest memory through GuestMemory
        // interfaces.
        // And the `used_index` is a two-byte naturally aligned field, so it won't cross the region
        // boundary and get_slice() shouldn't fail.
        let index_addr = self.avail_ring.unchecked_add(2);
        match vq_load_u16(mem.deref(), index_addr) {
            Ok(index) => {
                let offset = (4 + self.actual_size() * 8) as u64;
                let avail_event_addr = self.used_ring.unchecked_add(offset);
                if let Err(e) = vq_store_u16(mem.deref(), index, avail_event_addr) {
                    warn!("Can't update avail_event: {:?}", e);
                }
            }
            Err(e) => warn!("Invalid offset, {}", e),
        }
    }

    fn update_used_flag<M: GuestMemory>(&mut self, mem: &M, set: u16, clr: u16) {
        // Safe because we have validated the queue and access guest memory through GuestMemory
        // interfaces.
        // And the `used_index` is a two-byte naturally aligned field, so it won't cross the region
        // boundary and get_slice() shouldn't fail.
        let slice = mem
            .get_slice(self.used_ring, size_of::<u16>())
            .expect("invalid address for virtq_used.flags");
        let flag = slice.get_atomic_ref::<AtomicU16>(0).unwrap();
        let v = flag.load(Ordering::Relaxed);

        flag.store((v & !clr) | set, Ordering::Relaxed);
    }

    fn set_notification<M: GuestMemory>(&mut self, mem: &M, enable: bool) {
        self.event_notification_enabled = enable;
        if self.event_notification_enabled {
            if self.event_idx_enabled {
                self.update_avail_event(mem);
            } else {
                self.update_used_flag(mem, 0, VIRTQ_USED_F_NO_NOTIFY);
            }

            // This fence ensures that we observe the latest of virtq_avail once we publish
            // virtq_used.avail_event/virtq_used.flags.
            fence(Ordering::AcqRel);
        } else if !self.event_idx_enabled {
            self.update_used_flag(mem, VIRTQ_USED_F_NO_NOTIFY, 0);
        }
    }

    /// Enable notification events from the guest driver.
    #[inline]
    pub fn enable_notification<M: GuestMemory>(&mut self, mem: &M) {
        self.set_notification(mem, true);
    }

    /// Disable notification events from the guest driver.
    #[inline]
    pub fn disable_notification<M: GuestMemory>(&mut self, mem: &M) {
        self.set_notification(mem, false);
    }

    /// Return the value present in the used_event field of the avail ring.
    ///
    /// If the VIRTIO_F_EVENT_IDX feature bit is not negotiated, the flags field in the available
    /// ring offers a crude mechanism for the driver to inform the device that it doesn’t want
    /// interrupts when buffers are used. Otherwise virtq_avail.used_event is a more performant
    /// alternative where the driver specifies how far the device can progress before interrupting.
    ///
    /// Neither of these interrupt suppression methods are reliable, as they are not synchronized
    /// with the device, but they serve as useful optimizations. So we only ensure access to the
    /// virtq_avail.used_event is atomic, but do not need to synchronize with other memory accesses.
    fn get_used_event<M: GuestMemory>(&self, mem: &M) -> Option<Wrapping<u16>> {
        // Safe because we have validated the queue and access guest memory through GuestMemory
        // interfaces.
        // And the `used_index` is a two-byte naturally aligned field, so it won't cross the region
        // boundary and get_slice() shouldn't fail.
        let used_event_addr = self
            .avail_ring
            .unchecked_add((4 + self.actual_size() * 2) as u64);
        let used_event: Option<Wrapping<u16>> = match vq_load_u16(mem.deref(), used_event_addr) {
            Ok(u) => Some(Wrapping(u)),
            Err(_) => None,
        };

        used_event
    }

    /// Check whether a notification to the guest is needed.
    pub fn needs_notification<M: GuestMemory>(&mut self, mem: &M, used_idx: Wrapping<u16>) -> bool {
        let mut notify = true;

        // The VRING_AVAIL_F_NO_INTERRUPT flag isn't supported yet.
        if self.event_idx_enabled {
            if let Some(old_idx) = self.signalled_used.replace(used_idx) {
                if let Some(used_event) = self.get_used_event(mem) {
                    if (used_idx - used_event - Wrapping(1u16)) >= (used_idx - old_idx) {
                        notify = false;
                    }
                }
            }
        }

        notify
    }

    /// Goes back one position in the available descriptor chain offered by the driver.
    /// Rust does not support bidirectional iterators. This is the only way to revert the effect
    /// of an iterator increment on the queue.
    pub fn go_to_previous_position(&mut self) {
        self.next_avail -= Wrapping(1);
    }
}

fn vq_load_u16<M: GuestMemory>(mem: &M, addr: GuestAddress) -> Result<u16, Error> {
    let ret: u16 = match mem.get_slice(addr, size_of::<u16>()) {
        Ok(addr_slice) => addr_slice
            .get_atomic_ref::<AtomicU16>(0)
            .unwrap()
            .load(Ordering::Relaxed),
        Err(e) => {
            error!("Can't load guest addr (0x{:08x}): {:?}", addr.0, e);
            return Err(Error::GuestMemoryError);
        }
    };
    Ok(ret)
}

fn vq_store_u32<M: GuestMemory>(mem: &M, value: u32, addr: GuestAddress) -> Result<(), Error> {
    match mem.get_slice(addr, size_of::<u32>()) {
        Ok(addr_slice) => {
            addr_slice
                .get_atomic_ref::<AtomicU32>(0)
                .unwrap()
                .store(value, Ordering::Relaxed);
            return Ok(());
        }
        Err(e) => {
            error!("Can't store guest addr (0x{:08x}): {:?}", addr.0, e);
            return Err(Error::GuestMemoryError);
        }
    }
}

fn vq_store_u16<M: GuestMemory>(mem: &M, value: u16, addr: GuestAddress) -> Result<(), Error> {
    match mem.get_slice(addr, size_of::<u16>()) {
        Ok(addr_slice) => {
            addr_slice
                .get_atomic_ref::<AtomicU16>(0)
                .unwrap()
                .store(value, Ordering::Relaxed);
            return Ok(());
        }
        Err(e) => {
            error!("Can't store guest addr (0x{:08x}): {:?}", addr.0, e);
            return Err(Error::GuestMemoryError);
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    extern crate vm_memory;

    use std::marker::PhantomData;
    use std::mem;

    pub use super::*;
    use vm_memory::{
        Bytes, GuestAddress, GuestMemoryMmap, GuestMemoryRegion, MemoryRegionAddress,
        VolatileMemory, VolatileRef, VolatileSlice,
    };

    // Represents a virtio descriptor in guest memory.
    pub struct VirtqDesc<'a> {
        desc: VolatileSlice<'a>,
    }

    macro_rules! offset_of {
        ($ty:ty, $field:ident) => {
            unsafe { &(*(0 as *const $ty)).$field as *const _ as usize }
        };
    }

    impl<'a> VirtqDesc<'a> {
        fn new(dtable: &'a VolatileSlice<'a>, i: u16) -> Self {
            let desc = dtable
                .get_slice((i as usize) * Self::dtable_len(1), Self::dtable_len(1))
                .unwrap();
            VirtqDesc { desc }
        }

        pub fn addr(&self) -> VolatileRef<u64> {
            self.desc.get_ref(offset_of!(Descriptor, addr)).unwrap()
        }

        pub fn len(&self) -> VolatileRef<u32> {
            self.desc.get_ref(offset_of!(Descriptor, len)).unwrap()
        }

        pub fn flags(&self) -> VolatileRef<u16> {
            self.desc.get_ref(offset_of!(Descriptor, flags)).unwrap()
        }

        pub fn next(&self) -> VolatileRef<u16> {
            self.desc.get_ref(offset_of!(Descriptor, next)).unwrap()
        }

        pub fn set(&self, addr: u64, len: u32, flags: u16, next: u16) {
            self.addr().store(addr);
            self.len().store(len);
            self.flags().store(flags);
            self.next().store(next);
        }

        fn dtable_len(nelem: u16) -> usize {
            16 * nelem as usize
        }
    }

    // Represents a virtio queue ring. The only difference between the used and available rings,
    // is the ring element type.
    pub struct VirtqRing<'a, T> {
        ring: VolatileSlice<'a>,
        start: GuestAddress,
        qsize: u16,
        _marker: PhantomData<*const T>,
    }

    impl<'a, T> VirtqRing<'a, T>
    where
        T: vm_memory::ByteValued,
    {
        fn new(
            start: GuestAddress,
            mem: &'a GuestMemoryMmap,
            qsize: u16,
            alignment: GuestUsize,
        ) -> Self {
            assert_eq!(start.0 & (alignment - 1), 0);

            let (region, addr) = mem.to_region_addr(start).unwrap();
            let size = Self::ring_len(qsize);
            let ring = region.get_slice(addr, size).unwrap();

            let result = VirtqRing {
                ring,
                start,
                qsize,
                _marker: PhantomData,
            };

            result.flags().store(0);
            result.idx().store(0);
            result.event().store(0);
            result
        }

        pub fn start(&self) -> GuestAddress {
            self.start
        }

        pub fn end(&self) -> GuestAddress {
            self.start.unchecked_add(self.ring.len() as GuestUsize)
        }

        pub fn flags(&self) -> VolatileRef<u16> {
            self.ring.get_ref(0).unwrap()
        }

        pub fn idx(&self) -> VolatileRef<u16> {
            self.ring.get_ref(2).unwrap()
        }

        fn ring_offset(i: u16) -> usize {
            4 + mem::size_of::<T>() * (i as usize)
        }

        pub fn ring(&self, i: u16) -> VolatileRef<T> {
            assert!(i < self.qsize);
            self.ring.get_ref(Self::ring_offset(i)).unwrap()
        }

        pub fn event(&self) -> VolatileRef<u16> {
            self.ring.get_ref(Self::ring_offset(self.qsize)).unwrap()
        }

        fn ring_len(qsize: u16) -> usize {
            Self::ring_offset(qsize) + 2
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct VirtqUsedElem {
        pub id: u32,
        pub len: u32,
    }

    unsafe impl vm_memory::ByteValued for VirtqUsedElem {}

    pub type VirtqAvail<'a> = VirtqRing<'a, u16>;
    pub type VirtqUsed<'a> = VirtqRing<'a, VirtqUsedElem>;

    trait GuestAddressExt {
        fn align_up(&self, x: GuestUsize) -> GuestAddress;
    }
    impl GuestAddressExt for GuestAddress {
        fn align_up(&self, x: GuestUsize) -> GuestAddress {
            return Self((self.0 + (x - 1)) & !(x - 1));
        }
    }

    pub struct VirtQueue<'a> {
        start: GuestAddress,
        dtable: VolatileSlice<'a>,
        avail: VirtqAvail<'a>,
        used: VirtqUsed<'a>,
    }

    impl<'a> VirtQueue<'a> {
        // We try to make sure things are aligned properly :-s
        pub fn new(start: GuestAddress, mem: &'a GuestMemoryMmap, qsize: u16) -> Self {
            // power of 2?
            assert!(qsize > 0 && qsize & (qsize - 1) == 0);

            let (region, addr) = mem.to_region_addr(start).unwrap();
            let dtable = region
                .get_slice(addr, VirtqDesc::dtable_len(qsize))
                .unwrap();

            const AVAIL_ALIGN: GuestUsize = 2;

            let avail_addr = start
                .unchecked_add(VirtqDesc::dtable_len(qsize) as GuestUsize)
                .align_up(AVAIL_ALIGN);
            let avail = VirtqAvail::new(avail_addr, mem, qsize, AVAIL_ALIGN);

            const USED_ALIGN: GuestUsize = 4;

            let used_addr = avail.end().align_up(USED_ALIGN);
            let used = VirtqUsed::new(used_addr, mem, qsize, USED_ALIGN);

            VirtQueue {
                start,
                dtable,
                avail,
                used,
            }
        }

        fn size(&self) -> u16 {
            (self.dtable.len() / VirtqDesc::dtable_len(1)) as u16
        }

        fn dtable(&self, i: u16) -> VirtqDesc {
            VirtqDesc::new(&self.dtable, i)
        }

        fn dtable_start(&self) -> GuestAddress {
            self.start
        }

        fn avail_start(&self) -> GuestAddress {
            self.avail.start()
        }

        fn used_start(&self) -> GuestAddress {
            self.used.start()
        }

        // Creates a new Queue, using the underlying memory regions represented by the VirtQueue.
        pub fn create_queue(&self) -> Queue {
            let mut q = Queue::new(self.size());

            q.size = self.size();
            q.ready = true;
            q.desc_table = self.dtable_start();
            q.avail_ring = self.avail_start();
            q.used_ring = self.used_start();

            q
        }

        pub fn start(&self) -> GuestAddress {
            self.dtable_start()
        }

        pub fn end(&self) -> GuestAddress {
            self.used.end()
        }
    }

    #[test]
    pub fn test_offset() {
        assert_eq!(offset_of!(Descriptor, addr), 0);
        assert_eq!(offset_of!(Descriptor, len), 8);
        assert_eq!(offset_of!(Descriptor, flags), 12);
        assert_eq!(offset_of!(Descriptor, next), 14);
    }

    #[test]
    fn test_checked_new_descriptor_chain() {
        let m = &GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap();
        let vq = VirtQueue::new(GuestAddress(0), m, 16);

        assert!(vq.end().0 < 0x1000);

        // index >= queue_size
        assert!(DescriptorChain::<GuestMemoryMmap>::checked_new(m, vq.start(), 16, 16).is_none());

        // desc_table address is way off
        assert!(DescriptorChain::<GuestMemoryMmap>::checked_new(
            m,
            GuestAddress(0x00ff_ffff_ffff),
            16,
            0
        )
        .is_none());

        // the addr field of the descriptor is way off
        vq.dtable(0).addr().store(0x0fff_ffff_ffff);
        assert!(DescriptorChain::<GuestMemoryMmap>::checked_new(m, vq.start(), 16, 0).is_none());

        // let's create some invalid chains

        {
            // the addr field of the desc is ok now
            vq.dtable(0).addr().store(0x1000);
            // ...but the length is too large
            vq.dtable(0).len().store(0xffff_ffff);
            assert!(
                DescriptorChain::<GuestMemoryMmap>::checked_new(m, vq.start(), 16, 0).is_none()
            );
        }

        {
            // the first desc has a normal len now, and the next_descriptor flag is set
            vq.dtable(0).len().store(0x1000);
            vq.dtable(0).flags().store(VIRTQ_DESC_F_NEXT);
            //..but the the index of the next descriptor is too large
            vq.dtable(0).next().store(16);

            assert!(
                DescriptorChain::<GuestMemoryMmap>::checked_new(m, vq.start(), 16, 0).is_none()
            );
        }

        // finally, let's test an ok chain

        {
            vq.dtable(0).next().store(1);
            vq.dtable(1).set(0x2000, 0x1000, 0, 0);

            let mut c =
                DescriptorChain::<GuestMemoryMmap>::checked_new(m, vq.start(), 16, 0).unwrap();

            assert_eq!(
                c.memory() as *const GuestMemoryMmap,
                m as *const GuestMemoryMmap
            );
            assert_eq!(c.desc_table, vq.dtable_start());
            assert_eq!(c.queue_size, 16);
            assert_eq!(c.ttl, c.queue_size);
            let desc = c.next().unwrap();
            assert_eq!(desc.addr(), GuestAddress(0x1000));
            assert_eq!(desc.len(), 0x1000);
            assert_eq!(desc.flags(), VIRTQ_DESC_F_NEXT);
            assert_eq!(desc.next, 1);

            assert!(c.next().is_some());
            assert!(c.next().is_none());
        }
    }

    #[test]
    fn test_checked_new_descriptor_chain_cross_mem_region() {
        let m = &GuestMemoryMmap::from_ranges(&[
            (GuestAddress(0), 0x1000),
            (GuestAddress(0x1000), 0x1000),
        ])
        .unwrap();

        // The whole descriptor table crosses guest memory boundary, it should ok.
        assert!(
            DescriptorChain::<GuestMemoryMmap>::checked_new(m, GuestAddress(0), 512, 1).is_some()
        );
    }

    #[test]
    fn test_new_from_indirect_descriptor() {
        let m = &GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap();
        let vq = VirtQueue::new(GuestAddress(0), m, 16);

        // create a chain with two descriptor pointing to an indirect tables
        let desc = vq.dtable(0);
        desc.set(0x1000, 0x1000, VIRTQ_DESC_F_INDIRECT | VIRTQ_DESC_F_NEXT, 1);
        let desc = vq.dtable(1);
        desc.set(0x2000, 0x1000, VIRTQ_DESC_F_INDIRECT | VIRTQ_DESC_F_NEXT, 2);
        let desc = vq.dtable(2);
        desc.set(0x3000, 0x1000, 0, 0);

        let mut c: DescriptorChain<GuestMemoryMmap> =
            DescriptorChain::checked_new(m, vq.start(), 16, 0).unwrap();
        assert!(c.is_indirect());

        let region = m.find_region(GuestAddress(0)).unwrap();
        let dtable = region
            .get_slice(MemoryRegionAddress(0x1000u64), VirtqDesc::dtable_len(4))
            .unwrap();
        // create an indirect table with 4 chained descriptors
        let mut indirect_table = Vec::with_capacity(4 as usize);
        for j in 0..4 {
            let desc = VirtqDesc::new(&dtable, j);
            if j < 3 {
                desc.set(0x1000, 0x1000, VIRTQ_DESC_F_NEXT, (j + 1) as u16);
            } else {
                desc.set(0x1000, 0x1000, 0, 0 as u16);
            }
            indirect_table.push(desc);
        }

        // create an indirect table with 1 chained descriptor
        let dtable2 = region
            .get_slice(MemoryRegionAddress(0x2000u64), VirtqDesc::dtable_len(1))
            .unwrap();
        let desc2 = VirtqDesc::new(&dtable2, 0);
        desc2.set(0x8000, 0x1000, 0, 0);

        assert_eq!(c.index(), 0);
        assert!(c.has_next());
        // try to iterate through the first indirect descriptor chain
        for j in 0..4 {
            let desc = c.next().unwrap();
            if j < 3 {
                assert_eq!(desc.flags(), VIRTQ_DESC_F_NEXT);
                assert_eq!(desc.next, j + 1);
                assert!(c.has_next());
                assert_eq!(c.index(), 0);
            }
        }

        // try to iterate through the second indirect descriptor chain
        assert!(c.has_next());
        let desc = c.next().unwrap();
        assert_eq!(desc.addr(), GuestAddress(0x8000));

        // back to the main descriptor chain
        assert!(c.has_next());
        let desc = c.next().unwrap();
        assert_eq!(desc.addr(), GuestAddress(0x3000));

        assert!(!c.has_next());
        assert!(c.next().is_none());
        assert!(!c.has_next());
        assert!(c.next().is_none());
        assert!(!c.has_next());
    }

    #[test]
    fn test_new_from_indirect_descriptor_err() {
        {
            let m = &GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap();
            let vq = VirtQueue::new(GuestAddress(0), m, 16);

            // create a chain with a descriptor pointing to an indirect table
            let desc = vq.dtable(0);
            desc.set(0x1001, 0x1000, VIRTQ_DESC_F_INDIRECT, 0);

            let c: DescriptorChain<GuestMemoryMmap> =
                DescriptorChain::checked_new(m, vq.start(), 16, 0).unwrap();
            assert!(c.is_indirect());

            assert!(c.new_from_indirect().is_err());
        }

        {
            let m = &GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap();
            let vq = VirtQueue::new(GuestAddress(0), m, 16);

            // create a chain with a descriptor pointing to an indirect table
            let desc = vq.dtable(0);
            desc.set(0x1000, 0x1001, VIRTQ_DESC_F_INDIRECT, 0);

            let c: DescriptorChain<GuestMemoryMmap> =
                DescriptorChain::checked_new(m, vq.start(), 16, 0).unwrap();
            assert!(c.is_indirect());

            assert!(c.new_from_indirect().is_err());
        }
    }

    #[test]
    fn test_queue_and_iterator() {
        let m = &GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap();
        let vq = VirtQueue::new(GuestAddress(0), m, 16);

        let mut q = vq.create_queue();

        // q is currently valid
        assert!(q.is_valid(m));

        // shouldn't be valid when not marked as ready
        q.ready = false;
        assert!(!q.is_valid(m));
        q.ready = true;

        // or when size > max_size
        q.size = q.max_size << 1;
        assert!(!q.is_valid(m));
        q.size = q.max_size;

        // or when size is 0
        q.size = 0;
        assert!(!q.is_valid(m));
        q.size = q.max_size;

        // or when size is not a power of 2
        q.size = 11;
        assert!(!q.is_valid(m));
        q.size = q.max_size;

        // or if the various addresses are off

        q.desc_table = GuestAddress(0xffff_ffff);
        assert!(!q.is_valid(m));
        q.desc_table = GuestAddress(0x1001);
        assert!(!q.is_valid(m));
        q.desc_table = vq.dtable_start();

        q.avail_ring = GuestAddress(0xffff_ffff);
        assert!(!q.is_valid(m));
        q.avail_ring = GuestAddress(0x1001);
        assert!(!q.is_valid(m));
        q.avail_ring = vq.avail_start();

        q.used_ring = GuestAddress(0xffff_ffff);
        assert!(!q.is_valid(m));
        q.used_ring = GuestAddress(0x1001);
        assert!(!q.is_valid(m));
        q.used_ring = vq.used_start();

        {
            // an invalid queue should return an iterator with no next
            q.ready = false;
            let mut i = q.iter(m);
            assert!(i.next().is_none());
        }

        q.ready = true;

        // now let's create two simple descriptor chains

        {
            for j in 0..5 {
                vq.dtable(j).set(
                    0x1000 * (j + 1) as u64,
                    0x1000,
                    VIRTQ_DESC_F_NEXT,
                    (j + 1) as u16,
                );
            }

            // the chains are (0, 1) and (2, 3, 4)
            vq.dtable(1).flags().store(0);
            vq.dtable(4).flags().store(0);
            vq.avail.ring(0).store(0);
            vq.avail.ring(1).store(2);
            vq.avail.idx().store(2);

            let mut i = q.iter(m);

            {
                let mut c = i.next().unwrap();
                assert_eq!(c.index(), 0);

                assert!(c.has_next());
                c.next().unwrap();
                assert!(c.has_next());
                assert!(c.next().is_some());
                assert!(!c.has_next());
                assert!(c.next().is_none());
                assert!(!c.has_next());
                assert_eq!(c.index(), 0);
            }

            {
                let mut c = i.next().unwrap();
                assert_eq!(c.index(), 2);

                assert!(c.has_next());
                c.next().unwrap();
                assert!(c.has_next());
                c.next().unwrap();
                assert!(c.has_next());
                c.next().unwrap();
                assert!(!c.has_next());
                assert!(c.next().is_none());
                assert!(!c.has_next());
                assert_eq!(c.index(), 2);
            }
        }

        // also test go_to_previous_position() works as expected
        {
            assert!(q.iter(m).next().is_none());
            q.go_to_previous_position();
            let mut c = q.iter(m).next().unwrap();
            c.next().unwrap();
            c.next().unwrap();
            c.next().unwrap();
            assert!(!c.has_next());
            assert!(c.next().is_none());
        }
    }

    #[test]
    fn test_descriptor_and_iterator() {
        let m = &GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap();
        let vq = VirtQueue::new(GuestAddress(0), m, 16);

        let mut q = vq.create_queue();

        // q is currently valid
        assert!(q.is_valid(m));

        for j in 0..7 {
            vq.dtable(j).set(
                0x1000 * (j + 1) as u64,
                0x1000,
                VIRTQ_DESC_F_NEXT,
                (j + 1) as u16,
            );
        }

        // the chains are (0, 1), (2, 3, 4) and (5, 6)
        vq.dtable(1).flags().store(0);
        vq.dtable(2)
            .flags()
            .store(VIRTQ_DESC_F_NEXT | VIRTQ_DESC_F_WRITE);
        vq.dtable(4).flags().store(VIRTQ_DESC_F_WRITE);
        vq.dtable(5)
            .flags()
            .store(VIRTQ_DESC_F_NEXT | VIRTQ_DESC_F_WRITE);
        vq.dtable(6).flags().store(0);
        vq.avail.ring(0).store(0);
        vq.avail.ring(1).store(2);
        vq.avail.ring(2).store(5);
        vq.avail.idx().store(3);

        let mut i = q.iter(m);

        {
            let c = i.next().unwrap();
            assert_eq!(c.index(), 0);

            let mut iter = c.into_iter();
            assert!(iter.next().is_some());
            assert!(iter.next().is_some());
            assert!(iter.next().is_none());
            assert!(iter.next().is_none());
        }

        {
            let c = i.next().unwrap();
            assert_eq!(c.index(), 2);

            let mut iter = c.writable();
            assert!(iter.next().is_some());
            assert!(iter.next().is_some());
            assert!(iter.next().is_none());
            assert!(iter.next().is_none());
        }

        {
            let c = i.next().unwrap();
            assert_eq!(c.index(), 5);

            let mut iter = c.readable();
            assert!(iter.next().is_some());
            assert!(iter.next().is_none());
            assert!(iter.next().is_none());
        }
    }

    #[test]
    fn test_add_used() {
        let m = &GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap();
        let vq = VirtQueue::new(GuestAddress(0), m, 16);

        let mut q = vq.create_queue();
        assert_eq!(vq.used.idx().load(), 0);

        //index too large
        assert!(q.add_used(m, 16, 0x1000).is_none());
        assert_eq!(vq.used.idx().load(), 0);

        //should be ok
        assert_eq!(q.add_used(m, 10, 0x1000).unwrap(), 1);
        assert_eq!(q.add_used(m, 11, 0x2000).unwrap(), 2);
        assert_eq!(vq.used.idx().load(), 2);
        let x = vq.used.ring(0).load();
        assert_eq!(x.id, 10);
        assert_eq!(x.len, 0x1000);
        let x = vq.used.ring(1).load();
        assert_eq!(x.id, 11);
        assert_eq!(x.len, 0x2000);
    }

    #[test]
    fn test_reset_queue() {
        let m = &GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap();
        let vq = VirtQueue::new(GuestAddress(0), m, 16);

        let mut q = vq.create_queue();
        q.size = 8;
        q.ready = true;
        q.reset();
        assert_eq!(q.size, 16);
        assert_eq!(q.ready, false);
    }

    #[test]
    fn test_needs_notification() {
        let m = &GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap();
        let vq = VirtQueue::new(GuestAddress(0), m, 16);
        let mut q = vq.create_queue();
        let avail_addr = vq.avail_start();

        // It should always return true when EVENT_IDX isn't enabled.
        assert_eq!(q.needs_notification(m, Wrapping(1)), true);
        assert_eq!(q.needs_notification(m, Wrapping(2)), true);
        assert_eq!(q.needs_notification(m, Wrapping(3)), true);
        assert_eq!(q.needs_notification(m, Wrapping(4)), true);
        assert_eq!(q.needs_notification(m, Wrapping(5)), true);

        vq_store_u16(m, 4, avail_addr.unchecked_add(4 + 16 * 2)).expect("write 4 to avail_addr");
        q.set_event_idx(true);
        assert_eq!(q.needs_notification(m, Wrapping(1)), true);
        assert_eq!(q.needs_notification(m, Wrapping(2)), false);
        assert_eq!(q.needs_notification(m, Wrapping(3)), false);
        assert_eq!(q.needs_notification(m, Wrapping(4)), false);
        assert_eq!(q.needs_notification(m, Wrapping(5)), true);
        assert_eq!(q.needs_notification(m, Wrapping(6)), false);
        assert_eq!(q.needs_notification(m, Wrapping(7)), false);

        vq_store_u16(m, 8, avail_addr.unchecked_add(4 + 16 * 2)).expect("write 8 to avail_addr");
        assert_eq!(q.needs_notification(m, Wrapping(11)), true);
        assert_eq!(q.needs_notification(m, Wrapping(12)), false);

        vq_store_u16(m, 15, avail_addr.unchecked_add(4 + 16 * 2)).expect("write 15 to avail_addr");
        assert_eq!(q.needs_notification(m, Wrapping(0)), true);
        assert_eq!(q.needs_notification(m, Wrapping(14)), false);
    }

    #[test]
    fn test_enable_disable_notification() {
        let m = GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap();
        let vq = VirtQueue::new(GuestAddress(0), &m, 16);
        let mut q = vq.create_queue();
        let used_addr = vq.used_start();

        assert_eq!(q.event_notification_enabled, true);
        assert_eq!(q.event_idx_enabled, false);

        q.enable_notification(&m);
        let v = m.read_obj::<u16>(used_addr).unwrap();
        assert_eq!(v, 0);

        q.disable_notification(&m);
        let v = m.read_obj::<u16>(used_addr).unwrap();
        assert_eq!(v, VIRTQ_USED_F_NO_NOTIFY);

        q.enable_notification(&m);
        let v = m.read_obj::<u16>(used_addr).unwrap();
        assert_eq!(v, 0);

        q.set_event_idx(true);
        let avail_addr = vq.avail_start();
        m.write_obj::<u16>(2, avail_addr.unchecked_add(2)).unwrap();

        q.enable_notification(&m);
        let v = m
            .read_obj::<u16>(used_addr.unchecked_add(4 + 8 * 16))
            .unwrap();
        assert_eq!(v, 2);

        q.disable_notification(&m);
        let v = m
            .read_obj::<u16>(used_addr.unchecked_add(4 + 8 * 16))
            .unwrap();
        assert_eq!(v, 2);

        m.write_obj::<u16>(8, avail_addr.unchecked_add(2)).unwrap();
        q.enable_notification(&m);
        let v = m
            .read_obj::<u16>(used_addr.unchecked_add(4 + 8 * 16))
            .unwrap();
        assert_eq!(v, 8);
    }
}
