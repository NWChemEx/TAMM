#ifndef TIMER_HPP
#define TIMER_HPP
#include <chrono>
#include <forward_list>
#include <thread>
#include <iostream>
#include <string>
#include <unordered_map>

namespace tamm::fastcc{

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
        std::forward_list<void*> ptrs;
        void* current_ptr = NULL;
        uint64_t size = 0;
        uint64_t base = 0;
    public:
        ManagedHeap(uint64_t size){
            this->size = size;
            this->current_ptr = calloc(size, 1);
            if(this->current_ptr == NULL){
                std::cerr << "Failed to allocate memory in the managed heap" << std::endl;
                exit(1);
            }
            this->base = 0;
        }
        ManagedHeap(): ManagedHeap(((uint64_t)(1))<<30){}
        ManagedHeap(const ManagedHeap&) = delete;
        ~ManagedHeap(){
            this->destroy();
        }
        void destroy(){
            for(auto ptr : ptrs){
                free(ptr);
            }
            free(current_ptr);
        }
        void* alloc(uint64_t requested_size){
            if(requested_size + base > size){
                //std::cout << "Requested size "<<requested_size<<" is larger than the heap size " <<size << std::endl;
                //std::cout << "Base is " << base << std::endl;
                //exit(1);
                ptrs.push_front(current_ptr);
                this->current_ptr = calloc(size, 1);
                if(current_ptr == NULL){
                    std::cerr << "Failed to re-allocate memory in the managed heap" << std::endl;
                    exit(1);
                }
                this->base = 0;
            }
            void* ret = (void*)((char*)this->current_ptr + base);
            base += requested_size;
            return ret;
        }
};

static ManagedHeap* tlocal_heaps;

static void init_heaps(int num_threads){
    tlocal_heaps = new ManagedHeap[num_threads];
}
static void destroy_heaps(int num_threads){
    for(int i = 0; i < num_threads; i++){
        tlocal_heaps[i].destroy();
    }
}

static void *my_calloc(uint64_t num_elts, uint64_t size_per_elt, int thread_id) {
        return tlocal_heaps[thread_id].alloc(num_elts * size_per_elt);
}
static void *my_malloc(uint64_t size, int thread_id) {
        return tlocal_heaps[thread_id].alloc(size);
}
}

#endif
