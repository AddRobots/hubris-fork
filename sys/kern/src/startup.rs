// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Kernel startup.

use crate::atomic::AtomicExt;
use crate::descs::{RegionAttributes, RegionDesc, TaskDesc, TaskFlags};
use crate::task::Task;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicBool, Ordering};

/// Tracks when there are _no_ mutable references to the task table floating
/// around in kernel code, indicating that we can produce a new one. This forms
/// a sort of ad-hoc Mutex around the task table.
///
/// Notice that this begins life initialized to `false`. This prevents use of
/// `with_task_table` et al before the kernel is properly started. We set it to
/// `true` late in `start_kernel`.
static TASK_TABLE_AVAIL: AtomicBool = AtomicBool::new(false);

/// The memory where the task table will be stored. This is sized in terms of
/// the generated `HUBRIS_TASK_COUNT` constant, and is uninitialized until
/// written later in this file.
#[link_section = ".uninit"]
static mut HUBRIS_TASK_TABLE_SPACE:
    MaybeUninit<[crate::task::Task; HUBRIS_TASK_COUNT]> = MaybeUninit::uninit();

pub const HUBRIS_FAULT_NOTIFICATION: u32 = 1;

// Diagnostic-only bisection helper (see `diag-uart-checkpoint` feature).
// Defined by the board's clock_stubs.c; safe to call anywhere pre-scheduler
// since it only touches raw UART registers.
#[cfg(feature = "diag-uart-checkpoint")]
extern "C" {
    fn uart_checkpoint(msg: *const u8);
}

#[cfg(feature = "diag-uart-checkpoint")]
macro_rules! diag_checkpoint {
    ($msg:expr) => {
        unsafe { uart_checkpoint(concat!($msg, "\r\n\0").as_ptr()) }
    };
}
#[cfg(not(feature = "diag-uart-checkpoint"))]
macro_rules! diag_checkpoint {
    ($msg:expr) => {};
}

/// The main kernel entry point.
///
/// We currently expect an application to provide its own `main`-equivalent
/// function, which does basic hardware setup and then calls this function.
///
/// Parameters:
///
/// - `tick_divisor`: a platform-specific way of converting "machine ticks" into
///   "kernel ticks." On ARM M-profile, this is CPU cycles per tick, where a
///   tick is typically a millisecond.
///
/// # Safety
///
/// This function has architecture-specific requirements for safe use -- on ARM,
/// for instance, it must be called from the main (interrupt) stack in
/// privileged mode.
///
/// This function may not be called reentrantly or from multiple cores.
pub unsafe fn start_kernel(tick_divisor: u32) -> ! {
    // Set our clock frequency so debuggers can find it as needed
    //
    // Safety: Due to our own safety contract, we're calling this a single time
    // early in boot, so it's ok.
    unsafe {
        crate::arch::set_clock_freq(tick_divisor);
    }
    diag_checkpoint!("SK1");

    // Grab references to all our statics.
    let task_descs = &HUBRIS_TASK_DESCS;
    // Safety: this reference will remain unique so long as the "only called
    // once per boot" contract on this function is upheld.
    let task_table =
        unsafe { &mut *core::ptr::addr_of_mut!(HUBRIS_TASK_TABLE_SPACE) };

    // Initialize our RAM data structures.

    // We currently just refer to the RegionDescs in Flash. No additional
    // preparation of those structures is required here. This will almost
    // certainly need to change in the future: we can save many cycles by (1)
    // storing them in an architecture-optimized format for this particular MPU,
    // and (2) moving them into RAM where random accesses don't imply wait
    // states.

    // Now, generate the task table.
    // Safety: MaybeUninit<[T]> -> [MaybeUninit<T>] is defined as safe.
    let task_table: &mut [MaybeUninit<Task>; HUBRIS_TASK_COUNT] =
        unsafe { &mut *(task_table as *mut _ as *mut _) };
    for (i, task) in task_table.iter_mut().enumerate() {
        task.write(Task::from_descriptor(&task_descs[i]));
    }
    diag_checkpoint!("SK2");

    // Safety: we have fully initialized this and can shed the uninit part.
    let task_table: &mut [Task; HUBRIS_TASK_COUNT] =
        unsafe { &mut *(task_table as *mut _ as *mut _) };

    // With that done, set up initial register state etc.
    for task in task_table.iter_mut() {
        crate::arch::reinitialize(task);
    }
    diag_checkpoint!("SK3");

    // Great! Pick our first task. We'll act like we're scheduling after the
    // last task, which will cause a scan from 0 on.
    let first_task = crate::task::select(task_table.len() - 1, task_table);
    diag_checkpoint!("SK4");

    crate::arch::apply_memory_protection(first_task);
    diag_checkpoint!("SK5");
    TASK_TABLE_AVAIL.store(true, Ordering::Release);
    diag_checkpoint!("SK6");
    crate::arch::start_first_task(tick_divisor, first_task)
}

/// Runs `body` with a reference to the task table.
///
/// To preserve uniqueness of the `&mut` reference passed into `body`, this
/// function will detect any attempts to call it recursively and panic.
pub(crate) fn with_task_table<R>(body: impl FnOnce(&mut [Task]) -> R) -> R {
    if !TASK_TABLE_AVAIL.swap_polyfill(false, Ordering::Acquire) {
        panic!(); // recursive use of with_task_table
    }
    let task_table: *mut MaybeUninit<[Task; HUBRIS_TASK_COUNT]> =
        core::ptr::addr_of_mut!(HUBRIS_TASK_TABLE_SPACE);
    // Pointer cast valid as MaybeUninit<[T; N]> and [MaybeUninit<T>; N] have
    // same in-memory representation. At the time of this writing
    // MaybeUninit::transpose is not yet stable.
    let task_table: *mut [MaybeUninit<Task>; HUBRIS_TASK_COUNT] =
        task_table as _;
    // This pointer cast is doing the equivalent of
    // MaybeUninit::array_assume_init, which at the time of this writing is not
    // stable.
    let task_table: *mut [Task; HUBRIS_TASK_COUNT] = task_table as _;
    // Safety: we have observed `TASK_TABLE_AVAIL` being true, which means the
    // task table is initialized (note that at reset it starts out true) and
    // that we're not already within a call to with_task_table. Thus, we can
    // produce a reference to the task table without aliasing, and we can be
    // confident that the memory it's pointing to is initialized and shed the
    // MaybeUninit.
    let task_table = unsafe { &mut *task_table };

    let r = body(task_table);

    TASK_TABLE_AVAIL.store(true, Ordering::Release);

    r
}

include!(concat!(env!("OUT_DIR"), "/kconfig.rs"));
