#include <vector>
#include <queue>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>

class ThreadPooling {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::condition_variable cv_task;
    std::condition_variable cv_done;
    std::mutex mtx;

    std::atomic<bool> stop;
    size_t activeTasks = 0;

public:
    ThreadPooling(size_t size) : stop(false) {
        for (size_t i = 0; i < size; ++i) {
            workers.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        cv_task.wait(lock, [this]() {
                            return stop || !tasks.empty();
                        });

                        if (stop && tasks.empty()) return;

                        task = std::move(tasks.front());
                        tasks.pop();
                        activeTasks++;
                    }

                    // Execute task outside lock
                    task();

                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        activeTasks--;
                        if (tasks.empty() && activeTasks == 0) {
                            cv_done.notify_all();
                        }
                    }
                }
            });
        }
    }

    ~ThreadPooling() {
        {
            std::unique_lock<std::mutex> lock(mtx);
            stop = true;
        }
        cv_task.notify_all();
        for (std::thread &worker : workers) {
            if (worker.joinable()) worker.join();
        }
    }

    template <class Function>
    void enqueue(Function&& task) {
        {
            std::unique_lock<std::mutex> lock(mtx);
            tasks.emplace(std::forward<Function>(task));
        }
        cv_task.notify_one();
    }

    void wait_for_all_tasks() {
        std::unique_lock<std::mutex> lock(mtx);
        cv_done.wait(lock, [this]() {
            return tasks.empty() && activeTasks == 0;
        });
    }
};