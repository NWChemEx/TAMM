#ifndef TAMM_TASK_ENGINE_HPP_
#define TAMM_TASK_ENGINE_HPP_

class TaskEngine {
    public:
        TaskEngine() {}
        TaskEngine(TaskEngine&&) = default;
        TaskEngine(const TaskEngine&) = default;
        TaskEngine& operator=(TaskEngine&&) = default;
        TaskEngine& operator=(const TaskEngine&) = default;
        ~TaskEngine() = default;

        // TBD: modify block access pairs to variadic template
        template<typename FD, typename ...Args>
            void submitTask(FD& fd, BlockAccessPairs block_access_pairs, Args... args) {
                Task *tsk = new TaskImpl<FD, Args...>(fd,block_access_pairs , args...);
                computeDependencies(tsk);
            }

//        void execute_all_tasks() {
//            while(!ready_queue_.empty())  {
//                //pop a task from queue and execute
//                Task* x = ready_queue_.back();
//                ready_queue_.pop();
//                x->execute(); 
//                release(x);          
//            }
//        }

        void release(Task* tsk);
    private:
        // Or use unique id for each data
        std::map<Key, BlockReadersWriter*> dependence_table;
        std::queue<Task*> ready_queue_;
        std::queue<Task*> pending_queue_;

        std::mutex ready_queue_mutex;
        void computeDependencies(Task *x);
};

#endif // TAMM_TASK_ENGINE_HPP_