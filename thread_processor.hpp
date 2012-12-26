#ifndef THREAD_PROCESSOR_HPP
#define THREAD_PROCESSOR_HPP

#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp> 
#include <boost/thread/condition.hpp> 

class ThreadProcessor {
	public:
		ThreadProcessor(int max_wait_for_job_sec = 200, int number_of_worker_threads= 0) :io_mutex_(), cond_(), initialized_(false), actors_(), number_messages_(0), done_(false), message_list_(), max_wait_for_job_sec_(max_wait_for_job_sec),number_of_worker_threads_(number_of_worker_threads) { 
			start_workers();
		}
		~ThreadProcessor() {
			done_ = true;
			cond_.notify_all();
			//actors_.interrupt_all();
			actors_.join_all();

		}

		typedef boost::function <void () > work_t;

		/**
		 * you can use this to schedule a task to run if you don't care to wait for it to complet
		 *
		 * func should be a function that takes no arguments
		 */
		void post(work_t func) {
				boost::mutex::scoped_lock lock(io_mutex_);
				message_list_.push_back(func);

				number_messages_++;
				cond_.notify_one();
		}
		
		int queued() const { //this is intensionally not locked

			return number_messages_;
		}
	private:

		bool getJob(work_t &job) {
			boost::mutex::scoped_lock lock(io_mutex_);
			while (false == done_ && queued() ==0) {
				boost::system_time tAbsoluteTime = boost::get_system_time() + boost::posix_time::seconds(max_wait_for_job_sec_);
				cond_.timed_wait(io_mutex_, tAbsoluteTime);
			}
			if (queued() ==0) 
				return false;
			job = message_list_.front();
			message_list_.pop_front();
			number_messages_--;
			return true;
		}
		void worker(int worker_id __attribute__((unused)) ) {
			while (!done_) {
				work_t job;
				bool success = getJob(job);	
				if (false == success)
					break;
				job();
			}
		}

		void start_workers() {
			if  (false == initialized_ ) {
				int number_of_threads= 0==number_of_worker_threads_ ? boost::thread::hardware_concurrency():number_of_worker_threads_ ;
				for (int i=0; i < number_of_threads; i++) {
					actors_.create_thread(boost::bind(&ThreadProcessor::worker,this, i));
				}
				initialized_=true;
			}
		}
		boost::mutex io_mutex_;
		boost::condition cond_;

		bool initialized_;
		boost::thread_group actors_;

		volatile int number_messages_;
		volatile bool done_;
	
		std::list< work_t >  message_list_;
		int max_wait_for_job_sec_;
		int number_of_worker_threads_;
	public :

};

/**
 * Use this to post a series of jobs to run in parallel and then wait for them to complete in the scheduling thread 
 */
class BatchTracker {
	public:
		BatchTracker(ThreadProcessor *threadProcessorp) :number_of_jobs_total(0),number_of_jobs_complete(0), cond_(), mutex() ,threadProcessorp_(threadProcessorp){}
		~BatchTracker() {};

		/** 
		 * post a function to run in parallel
		 * @func a function that takes no arguments to run in parallel, use a boost::bind to schedule with arguements
		 */
		void post(boost::function <void ()> func ) {
			threadProcessorp_->post(boost::bind(&BatchTracker::wrap, this, func));
			incJobCount();
		}
		/**
		 * @param seconds max number of seconds to wait for all jobs to complete
		 */
		bool wait_until_done(time_t seconds = 0) {
			if (number_of_jobs_complete == number_of_jobs_total)
				return true;
			return wait_until_done_locked(seconds);
		}
		/** 
		 * call this before re-using a batch tracker after calling wait_until_done
		 */
		void reset() {
			number_of_jobs_total = number_of_jobs_complete=0;
		}

		/** 
		 * returns number of time post was called since instansiation or reset() called
		 */
		int scheduled() const {
			return number_of_jobs_total; 
		}
		/** 
		 * number of jobs that have completed since instansiation or reset() called
		 */
		int complete() const {
			return number_of_jobs_complete; 
		}
	private:
		int incJobCount() {
			return number_of_jobs_total++;

		}
		void done() {
			boost::mutex::scoped_lock lock(mutex);
			number_of_jobs_complete++;
			if (number_of_jobs_complete == number_of_jobs_total)
				cond_.notify_one();
		}
		void wrap(boost::function <void ()> func) {
			func();	
			done();
		}
		bool wait_until_done_locked(time_t max_seconds) {
			boost::mutex::scoped_lock lock(mutex);
			time_t start, now =start= time(NULL);
			do {
				cond_.wait(mutex);
				if (number_of_jobs_complete != number_of_jobs_total)  {
					now = time(NULL);
				}
			}while  (number_of_jobs_complete != number_of_jobs_total && (max_seconds == 0  || ((now - start) < max_seconds))) ;
			return (number_of_jobs_complete == number_of_jobs_total);
		}
		volatile int number_of_jobs_total;
		volatile int number_of_jobs_complete;
		boost::condition cond_;
		boost::mutex mutex;
		ThreadProcessor *threadProcessorp_;
};
#endif
