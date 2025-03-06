#ifndef TIMER_HPP
#define TIMER_HPP
#include <chrono>
#include <thread>
#include <iostream>
#include <string>
#include <unordered_map>

class Timer {
private:
  std::unordered_map<std::string, float> func_timer;
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point end;
  std::string current_func = "";

public:
  void start_timer(std::string func_name) {
    if (current_func != "") {
      std::cerr << "Didn't end previous timer " << current_func
                << " before starting a new one" << std::endl;
      exit(1);
    }
    current_func = func_name;
    start = std::chrono::high_resolution_clock::now();
  }
  void end_timer(std::string func_name) {
    if (current_func != func_name) {
      std::cerr << "Trying to end previous timer " << current_func
                << " using a different one " << func_name << std::endl;
      exit(1);
    }
    end = std::chrono::high_resolution_clock::now();
    float this_time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    current_func = "";
    auto it = func_timer.find(func_name);
    if (it != func_timer.end())
      it->second += this_time;
    else
      func_timer[func_name] = this_time;
  }
  void print_all_times() {
    for (auto &p : func_timer) {
      std::cout << p.first << ", " << p.second << std::endl;
    }
  }
};

class ManagedHeap{
    private:
        void* ptr;
        uint64_t size = 0;
        uint64_t base = 0;
    public:
        ManagedHeap(uint64_t size){
            this->size = size;
            this->ptr = calloc(size, 1);
            if(this->ptr == NULL){
                std::cerr << "Failed to allocate memory in the managed heap" << std::endl;
                exit(1);
            }
            this->base = 0;
        }
        ManagedHeap(): ManagedHeap(((uint64_t)(1))<<33){}
        ManagedHeap(const ManagedHeap&) = delete;
        ~ManagedHeap(){
            free(ptr);
        }
        void* alloc(uint64_t requested_size){
            //TODO add linked list, can't just realloc because we have given out pointers
            if(requested_size + base > size){
                std::cerr << "Requested size "<<requested_size<<" is larger than the heap size " <<size << std::endl;
                std::cerr << "Base is " << base << std::endl;
                exit(1);
            }
            void* ret = (void*)((char*)ptr + base);
            base += requested_size;
            return ret;
        }
};

static ManagedHeap* tlocal_heaps;

static void init_heaps(int num_threads){
    tlocal_heaps = new ManagedHeap[num_threads];
}

static void *my_calloc(uint64_t num_elts, uint64_t size_per_elt, int thread_id) {
        return tlocal_heaps[thread_id].alloc(num_elts * size_per_elt);
}
static void *my_malloc(uint64_t size, int thread_id) {
        return tlocal_heaps[thread_id].alloc(size);
}

#endif
