#ifndef TAMM_EXECUTION_ENGINE_HPP_
#define TAMM_EXECUTION_ENGINE_HPP_

class RuntimeEngine {
public:
    RuntimeEngine() = default;
    RuntimeEngine(TaskEngine* taskengine);

    ~RuntimeEngine();
    void executeAllthreads(TaskEngine* taskengine);
    void executeAllthreads();

private:
    TaskEngine* te;
};

#endif // TAMM_EXECUTION_ENGINE_HPP_