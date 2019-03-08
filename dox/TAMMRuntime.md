TAMM Runtime

[[TOC]]

# Introduction

The purpose of the TAMM runtime engine is to enable automated communication management (including communication overlap and caching), memory management (including memory bounded execution), reordering and parallelization based on dependence anlaysis, and an abstraction for execution on multiple compute devices (e.g., CPU vs GPU).

The runtime engine operates on the following abstractions: tasks, data collections (primarily tensors),  data blocks.

Programmer submits tasks to the runtime engine, and then runtime manages data allocations/deallocations on different memories, data transfers, scheduling of tasks, communication-computation overlap. Tasks are submitted with a specific permission on data collections and request accesses to data blocks from data collections in various modes. The runtime schedules tasks when the task's dependences have been satisfied and data or buffer to which the task requested access are available in the desired mode.

Tasks are structured around permissions and access patterns on blocks of memory. Tasks are expressed at the level of labeled tensor and particular tensor blocks (IndexTensor(tensor, blockid)). The permissions determine which parts of tensors tasks are allowed to process, and access patterns inform the runtime which of the allowed blocks will actually be accessed in any given task. Tasks can be submitted recursively (including possible tail recursion), and the child tasks can request permissions that are a subset of parent permissions. Access patterns of a task have to be a subset of its permissions. The declared permissions allow the runtime to determine the dependencies between tasks and to execute them without conflicts, and the access patterns support efficient memory and communication management.

This document describes high-level ideas. Specific APIs are described in detail in the generated code documentation.

# Permissions and Access Modes

## Permissions

Permissions denote the blocks that a task can operate on and what operations can be performed. There are 3 types of permissions:

* **Read**,

* **Write**, and

* **Accumulate**.

A read+write permission is provided explicitly in the runtime for convenience, but it is equivalent to requesting read and write permissions separately.

Permissions define dependencies between tasks, that are then used to execute independent tasks in parallel. All block accesses in a task are checked against the permissions, and nested tasks cannot exceed the permissions of their parents. [This inclusive specification allows the runtime to guarantee sequential semantics while enabling dependence analysis at a coarse level]

Permissions can be requested for specific blocks of tensors, or for whole tensors or their slices. A permission to a particular block is specified by a tensor and a block id in the tensor, and access to a larger portion of a tensor is specified as a labeled tensor.

## Access Modes

To actually access a buffer, a task must request access to a buffer. Access requests must correspond to the appropriate permission that a task has, but not every permission requires access. The following access modes are supported:

* **Read**: Request read access to a block.

* **Write**: Request write access to a block. All elements in the block are assumed to be overwritten.

* **Read/Write**: Request read and writes to a block. First reads a portion of data block and then also writes the data block.

* **Accum**: Accumulate to a data block. The buffer is assumed to be initialized.

* **Temp**: A temporary local copy of a data block. It does not impact the tensor’s data block until explicitly requested to do so. Temp access does not conflict with any other modes and any number of simultaneous Temp access accesses are allowed. Temporary blocks can be read, written, or accumulated according to task permissions.

Access modes inform the runtime ahead of time which buffers will be accessed in what way, but the actual creation of local buffers is performed by access functions corresponding to the access modes. Access modes are used by the runtime system to prepare data in the appropriate memory before the task execution starts. It allows to achieve maximum communication-computation overlap. The local buffers are populated with tensor data for Read and Read/Write accesses. For all other access patterns, the block must be populated by the user.

Any of the modifying access (Write, Read/Write, and Accum) can be requested in a cancellable mode. In cancellable mode, the runtime ensures that memory accessed by a task is not directly mapped to data in a tensor, allowing to discard changes. In a non-cancellable mode, changes to local block buffers may result in immediate and irrevocable changes to a tensor.

# Runtime Engine and Task Submission

Runtime Engine executes tasks and manages block buffers memory. The runtime engine maintains all the state necessary to perform these tasks. A runtime engine is associated with execution context and it is accessed in the following way:

```c++
ExecutionContext ec;

RuntimeEngine *re = ec.re();
```

A task, once it begins execution, does not block. If a task needs access to a block, ideally it does not just get a block, especially a ready copy. it would request access and submit a continuation. All such access requests are processed and the blocks are ensured to be in ready state before a task is executed.

## Task Submission

`submitTask`: This function is used to submit tasks to the runtime system. It requires to specify at least one execution lambda, access modes and permissions on the required data blocks, and any additional arguments that a task requires. For example:

`submitTask(cpuLambda, ReadPermission(A), WritePermission(B), ReadAccess(C), AccumAccess(D))`

Once submitted, a task is executed when no other tasks with overlapping permissions execute. The task is guaranteed to execute to completion. While the buffer access functions may result in data movement, they are guaranteed not to block. Buffers requested at task submission time with access mode may be optimized by the runtime and may be prefetched so no data movement occurs during task execution. While a task can access blocks that it did not explicitly request with an access mode, such accesses may be highly inefficient because the runtime is not given a chance to optimize them. Because of that, if a task contains branches or other control flow mechanisms, the branches should be submitted as continuation tasks with appropriate access requests.

Currently, the task submission interface takes a single function. However, this is a point of extension for future execution on heterogeneous hardware. When executing on heterogeneous hardware, the task may have to be specialized into multiple code versions, one for every type of the hardware (e.g., CPU function, CUDA function, OpenCL function). One possible implementation of such interface will involve a function wrapper that can be instantiated with versions of code for different devices:

```c++
Class FunctionDefinition; // one lambda for each architecture

FunctionDefinition lambdas;

Lambdas.cpu = [...](...){...}; // Some CPU code

Lambdas.gpu = gpuFunction; // Some GPU code

submitTask(lambdas, ReadPermission(Block1), WriteAccess(Block2));
```

## RuntimeContext

First argument of each lambda function is  runtime context. It is created by the runtime system when scheduling decision for a task is made. Runtime context contains information about the memory node of the worker where task is scheduled for execution and details of  present task. Runtime context allows  a task to retrieve updated copy of buffer on local memory (it may be device memory). It also allows tasks to submit nested tasks while maintaining dependencies among  other tasks.

# Blocks and Buffers

A tensor consists of blocks of memory that can only be accessed as a whole. These memory blocks can only be accessed through the runtime interfaces (explicit access is possible but its effects are implementation-specific when combined with the runtime-engine access). The runtime provides a local representation of memory blocks through block buffers (the BlockBuffer class). Block buffers are regular C++ objects supporting all expected semantics of such objects. In the same manner that the block buffers are obtained from the runtime, they must be also returned to the runtime explicitly with a request for a particular effect. For, example, simply requesting a block for writing does not guarantee that a write is completed, and for non-cancellable block, effects on the tensor are undefined if the block is not written explicitly. So, while locally block buffers are regular C++ objects, there are some tensor semantics to keep in mind:

* A non-cancellable write or accumulate block obtained from the buffer may be copied and its values can be compared to other blocks, but all the copies are deep copies. Any deep copy of a non-cancellable block may be used locally, but it cannot be committed back to the tensor. This limitation is dictated by potential optimizations that create a block buffer from memory that is directly contained in a tensor.

* A design decision has been made not to perform any operations in destructors of blocks other than freeing memory. So, for operations on a block to take effect, they must be explicitly committed using the runtime interface. There are four main reasons for this choice:

    * Committing memory effects can cause exceptions, and we require that destructors do not throw exceptions (exception safety).

    * While permissions are general such as write, a block buffer can be written using different kinds of operations that need to be explicitly specified. While some buffers could be potentially written automatically by the runtime, having a dual interface for different situations could be confusing.

    * In some situations (e.g., cancellable write permission), there may be multiple local copies (block buffers) of a block. In such situations, the programmer must choose explicitly which block buffer to commit (or not to commit any of the copies).

    * Allowing explicit commits of memory effects can be important for optimization by allowing better overlap between computation and communication and more timely memory management actions by the runtime.

## Retrieving buffers

A task can retrieve buffers (on the memory node where the  task is scheduled) using tensor and blockid. When a buffer is created, it has the the information about corresponding (tensor, blockid).

`get_buf_tmp` : a temporary buffer is returned (can be filled with initial value).

`get_buf_read`:  an updated copy of (tensor,blockid) is returned.

`get_buf_write`: an uninitialized buffer for (tensor,blockid) is returned (can be filled with initial value).

`get_buf_read_write`: an updated buffer containing a copy of (tensor,blockid) is returned and this buffer is later modified.

`get_buf_cwrite`, `get_buf_creadwrite`: these functions request cancellable versions of writable buffers. A cancellable buffer can be modified, and it is guaranteed that the modifications do not take place until explicitly committed. Also, cancellable buffers can be committed from any copy of the buffer while non-cancellable buffers must be committed from the buffer obtained from the runtime.

If some buffers are accessed in a task which are not indicated using access modes, then runtime synchronously prepares buffers for those accesses. Other buffers are prepared before the task execution starts. Because of that, buffers should be requested in task submission whenever possible. In practice, this means that tasks have to often be written in a continuation style where the next task requests the buffers necessary for the continuation.

## Pushing buffers

Block buffers must be explicitly released to commit memory effects. Currently, there are two interfaces that correspond to functions on tensor blocks. This list will be expanded as necessary to reflect all the possible operations.

`release_put` : This updates the global (tensor,blockid)  tensor with the current content of buffer.

`release_add`: This appends the current content of buffer to global (tensor,blockid) tensor.

There is also a release function to release a buffer without committing the effects (invalid to call on non-cancellable buffers).

### An example from AddOp

```c++
auto cpulambda = [=](RuntimeEngine::RuntimeContext rc_recursive) {

        BlockBuffer lbf = rc_recursive.get_buf_tmp(ltensor, lblockid);

        BlockBuffer rbf = rc_recursive.get_buf_read(rtensor, rblockid);

        const auto& ldims = lhs_.tensor().block_dims(lblockid);

        const auto& rdims = rhs_.tensor().block_dims(rblockid);

        SizeVec ldims_sz, rdims_sz;

        for(const auto v : ldims) { ldims_sz.push_back(v); }

        for(const auto v : rdims) { rdims_sz.push_back(v); }

        kernels::assign(lbf.data(), ldims_sz,

                lhs_int_labels_, alpha_, rbf.data(),

                rdims_sz, rhs_int_labels_,

                is_assign_);

        lbf.release_put();

        };

re.submitTask(cpulambda,

        TempAccess(IndexedTensor{ltensor, lblockid}),

        WritePermission(IndexedTensor{ltensor, lblockid}),

        ReadAccess(IndexedTensor{rtensor, rblockid}));
```

# **Ongoing Work**

## **Directed acyclic graph construction**

Based on different permissions specified in tasks, runtime constructs a dependency graph where nodes represent tasks and edges represent (data) dependencies among tasks.

## **ReadyQueue**

Runtime system has a queue which contains set of all ready tasks. Each idle worker pops a task from this queue.

## **Execution model**

Each worker pops a task from ready queue. If a worker is a CPU (resp. CUDA) worker then it executes cpulambda (resp. cudalambda).

## **After task execution**

When a task completes its execution, it releases the dependencies of dependent tasks. When all dependencies of a task is met, it is added to ready queue.

## **Task Scheduler**

This component is mainly responsible for allocation of tasks on efficient resources and communication-computation overlap. This should be flexible enough such that programmers can implement different types of scheduling heuristics (decision based on load on resources, affinity towards resources, minimization of data transfers, etc.) easily.

In beginning, we start with a simple heuristic. Each idle worker pops a task and execute it.

## Other Symbols related to Memory Operations

There are more memory operations we are considering to allow more optimizations:

* `CANCEL`: This cancels the an access to that data block without performing the stated operation. Canceling accesses is important if a user requested modify access (READ, RW, ACC) and does not end up doing the operation.

* `EVICT`: Evict a block. To be used by internal runtime managing the memory block.

* `FLUSH`: Flush a local changes in a block to remote. This is to be used to ensure ordering and completion semantics.

## Permissions Checks

A task has effects and accesses. Any access mode requested should be compatible with the effects it is permitted to request. Below is the table of allowed accesses for a given effect/permission.

Effect specification (or permission) for a block can be: ReadPermission, WritePermission, ReadWritePermission, AccumPermission

<table>
  <tr>
    <td></td>
    <td>Effect</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Permissible Operation</td>
    <td>ReadPermission</td>
    <td>WritePermission</td>
    <td>ReadWritePermission</td>
    <td>AccumPermission</td>
  </tr>
  <tr>
    <td>READ</td>
    <td>ok</td>
    <td>ok</td>
    <td>ok</td>
    <td></td>
  </tr>
  <tr>
    <td>WRITE</td>
    <td></td>
    <td>ok</td>
    <td>ok</td>
    <td></td>
  </tr>
  <tr>
    <td>RW</td>
    <td></td>
    <td>ok</td>
    <td>ok</td>
    <td></td>
  </tr>
  <tr>
    <td>ACC</td>
    <td></td>
    <td>ok</td>
    <td>ok</td>
    <td>ok</td>
  </tr>
  <tr>
    <td>TMP</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>RELEASE</td>
    <td>ok</td>
    <td>ok</td>
    <td>ok</td>
    <td>ok</td>
  </tr>
  <tr>
    <td>CANCEL</td>
    <td>ok</td>
    <td>ok</td>
    <td>ok</td>
    <td>ok</td>
  </tr>
  <tr>
    <td>EVICT</td>
    <td>*</td>
    <td>*</td>
    <td>*</td>
    <td>*</td>
  </tr>
  <tr>
    <td>FLUSH</td>
    <td>ok</td>
    <td>ok</td>
    <td>ok</td>
    <td>ok</td>
  </tr>
  <tr>
    <td>DONE_
WAITING</td>
    <td>*</td>
    <td>*</td>
    <td></td>
    <td>*</td>
  </tr>
</table>


EVICT and DONE_WAITING are runtime internal access requests and are not affected by the access modes.

## Submission FSM; Dependence Analysis; DAG Construction

The following table shows the transition rules and set of actions for different permissions. A DAG can be constructed from a set of submitted tasks following the actions listed in the table. The above described permissions produce 3 different types of effect: _READ, _WRITE, _ACC, therefore at least 3 states are required to analyze effect of different permissions. One extra state is required to denote unaccessed state, similar to start state.

State for a block can be: _READ, _WRITE, _ACC, _UNACCESSED

<table>
  <tr>
    <td></td>
    <td>Permission on data block A in task T</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Effect state</td>
    <td>Read</td>
    <td>Write</td>
    <td>Accum</td>
  </tr>
  <tr>
    <td>_UNACCESSED</td>
    <td>DT(A,R) = T /_READ</td>
    <td>DT(A, W)= T /_WRITE</td>
    <td>DT(A,ACC) = T /_ACC</td>
  </tr>
  <tr>
    <td>_READ</td>
    <td>DT(A,R) = DT(A,R) U {T} /_READ</td>
    <td>Remove(DT(A,R)), DT(A,W)  = T /_WRITE</td>
    <td>Remove(DT(A,R)), DT(A,ACC)= T /_ACC</td>
  </tr>
  <tr>
    <td>_WRITE</td>
    <td>Remove(DT(A,W)), DT(A,R)  =T /_READ</td>
    <td>DT(A,W)  = T /_WRITE</td>
    <td>Remove(DT(A,W)), DT(A,ACC)= T /_ACC</td>
  </tr>
  <tr>
    <td>_ACC</td>
    <td>Remove(DT(A, ACC)), DT(A, W) = T /_Read</td>
    <td>Remove(DT(A, ACC)), DT(A, W) = T /_WRITE</td>
    <td>DT(A,ACC)= DT(A,ACC) U {T} /_ACC</td>
  </tr>
</table>


Remove operations indicate edges between DT(A,*) and task T.  

The above table ensures that the task T has edges from all conflicting accesses.

## Execution FSM; Multiple Memory Units (Managing buffers in multiple memory units) 

Block states: _xxx denotes state of a data block in a memory unit. 

* _Unalloc:  No memory block is allocated in the current memory unit for the data block.

* _UNI:  An uninitialized memory block is assigned for the data block in the current memory unit.

* _Ready: This indicates that the content of  the data block in current memory unit and main memory unit is same.

* _Dirty:  This indicates that the data block may have changed in current memory unit compared to main memory unit.

* _DirtyAcc: This indicates that the data block has partial computed result for a user level operation.

NOTES

* When an access mode is requested, it is expected the invoked performs that operation. LATER: is a user wishes to skip an operation, he/she needs to explicitly cancel that access.

* Every call to access is counted and must be correspondingly released (implicitly or explicitly)

* The transition system ensures that the runtime (on a single thread/rank) maintains at most one buffer per block id permemory node (excluding the TMP accesses which are unbounded)

* Preconditions are specified for debugging and sanity checks. 

* Put,Get, and AccAll are communication operations.

* RC denotes reference count, the number of outstanding references (access without a corresponding release) to a data block.

* DONE_WAITING is an internal state request to be used by the runtime internals. The user cannot request this access.

Transitions: x / y / z, where x, y, and z are comma-separated lists. x denotes the preconditions for the transition, y denotes the list of actions, z denotes the new state for the block.

<table>
  <tr>
    <td></td>
    <td>Block State</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Operation</td>
    <td>_Unalloc</td>
    <td>_UNI</td>
    <td>_Ready</td>
    <td>_Dirty</td>
    <td>_DirtyAcc</td>
  </tr>
  <tr>
    <td>READ</td>
    <td>-- /
Allocate,
RC = 0 /
 _UNI</td>
    <td>RC == 0,
!pending / Nb(Get,blk,tsk),
RC = 1 /
_Ready</td>
    <td>-- /
NoOp,
RC += 1 /
_Ready</td>
    <td>RC == 0 /
Nb(Put,blk,tsk),
RC = 1 /
_Ready</td>
    <td>RC == 0 /
Nb(AccAll,blk,tsk),
Nb(Get,blk,tsk),
RC = 1 /
_Ready</td>
  </tr>
  <tr>
    <td>WRITE</td>
    <td>-- /
Allocate,
RC = 0 /
 _UNI</td>
    <td>RC == 0 /
NoOp,
RC = 1 /
_Dirty</td>
    <td>RC == 0 /
NoOp,
RC = 1 /
_Dirty</td>
    <td>RC == 0 /
NoOp,
RC = 1 /
_Dirty</td>
    <td>RC == 0 /
Nb(AccAll,blk,tsk),
RC = 1 /
_Dirty</td>
  </tr>
  <tr>
    <td>RW</td>
    <td>-- /
Allocate,
RC = 0 /
 _UNI</td>
    <td>RC == 0,
!pending /
Nb(Get,blk,tsk),
RC = 1 /
_Dirty</td>
    <td>RC == 0 /
NoOp,
RC = 1 /
_Dirty</td>
    <td>RC == 0 /
NoOp,
RC = 1 /
_Dirty</td>
    <td>RC == 0/
Nb(AccAll,blk,tsk),
Nb(Get,blk,tsk),
RC = 1 /
_Dirty</td>
  </tr>
  <tr>
    <td>ACC</td>
    <td>-- /
Allocate,
RC = 0 /
 _UNI</td>
    <td>RC == 0 /
NoOp,
RC  = 1  /
_DirtyAcc</td>
    <td>RC == 0/
NoOp,
RC = 1 /
_DirtyAcc</td>
    <td>RC == 0/
Nb(Put,blk,tsk),
RC = 1 /
_DirtyAcc</td>
    <td>RC==0 /
NoOp,
RC = 1 /
_DirtyAcc</td>
  </tr>
  <tr>
    <td>TMP</td>
    <td>IGNORE</td>
    <td>IGNORE</td>
    <td>IGNORE</td>
    <td>IGNORE</td>
    <td>IGNORE</td>
  </tr>
  <tr>
    <td>RELEASE</td>
    <td>ERR</td>
    <td>ERR</td>
    <td>RC > 0/
NoOp,
RC -= 1/
_Ready</td>
    <td>RC == 1 /
NoOp,
RC -= 1 /
_Dirty</td>
    <td>RC > 0/
NoOp,
RC -= 1/
_DirtyAcc</td>
  </tr>
  <tr>
    <td>CANCEL</td>
    <td>ERR</td>
    <td>ERR</td>
    <td>RC > 0/
NoOp,
RC -= 1/
_Ready</td>
    <td>RC == 1 /
NoOp,
RC -= 1 /
_Dirty</td>
    <td>RC > 0/
NoOp,
RC -= 1/
_DirtyAcc</td>
  </tr>
  <tr>
    <td>EVICT</td>
    <td>-- /
NoOp /
--</td>
    <td>RC == 0 /
Dealloc /
_Unalloc
</td>
    <td>RC == 0 /
Dealloc /
_Unalloc</td>
    <td>RC == 0 /
Nb(Put,blk,tsk) /
_Ready</td>
    <td>RC == 0 /
Nb(AccAll,blk,tsk)/ _UNI</td>
  </tr>
  <tr>
    <td>FLUSH</td>
    <td>--/
NoOp/
--</td>
    <td>--/
NoOp
--</td>
    <td>--/
NoOp/
--</td>
    <td>RC == 0/
Nb(Put,blk,tsk) /
_Ready</td>
    <td>RC == 0 /
Nb(AccAll,blk,tsk)/ _UNI</td>
  </tr>
</table>

