import sys
import numpy as np
import random 
from queue import PriorityQueue
import time

class Event:
    def __init__(self, time, type, server):
        self.time = time
        self.server = server   
        self.type = type
    
    def __lt__(self, other):
        # PriorityQueue uses this method to determine order
        # Lower time values have higher priority
        return self.time < other.time

class Departure(Event):
    def __init__(self, time, type, server, service_time, wait_time = 0):
        super().__init__(time, type, server)
        self.service_time = service_time
        self.wait_time = wait_time

        
class Server:
    def __init__(self, mu, max_queue_size, event_queue):
        self.mu = mu
        self.max_queue_size = max_queue_size
        self.event_queue = event_queue
        self.server_queue = []
        self.busy = False
        self.customers_served = 0
        self.customers_rejected = 0

        self.total_wait_time = 0
        self.total_service_time = 0
        self.last_departure_time = 0 
        
    def handle_arrival(self, current_time):
        if not self.busy:
            # Server is idle - start service immediately
            self.busy = True
            service_time = random.expovariate(self.mu)
            departure_time = current_time + service_time
            self.event_queue.put(Departure(departure_time, 'DEPARTURE', self, service_time))
        elif len(self.server_queue) < self.max_queue_size:
            # Server busy but queue has space - add to queue
            self.server_queue.append(current_time)
        else:
            # Queue is full - reject customer
            self.customers_rejected += 1
            
    def handle_departure(self,current_time, departure_event: Departure):
        #update server stats
        self.last_departure_time = current_time
        self.customers_served += 1
        self.total_service_time += departure_event.service_time
        self.total_wait_time += departure_event.wait_time
        #handle next customer
        if self.server_queue:
            arrival_time = self.server_queue.pop(0)
            wait_time = current_time - arrival_time
            service_time = random.expovariate(self.mu)
            departure_time = current_time + service_time  
            self.event_queue.put(Departure(departure_time, 'DEPARTURE', self, service_time, wait_time))
        else:
            self.busy = False
            
    def average_wait_time(self):
        if self.customers_served == 0:
            return 0
        return self.total_wait_time / self.customers_served
    
    def average_service_time(self):
        if self.customers_served == 0:
            return 0
        return self.total_service_time / self.customers_served

        
class loadBalancer:
    def __init__(self, lambda_, mu, N):
        self.lambda_ = lambda_
        self.mu = mu
        self.N = N
        self.events = PriorityQueue()
        self.server = Server(mu, N - 1, self.events) 

    def print_comprehensive_results(self):
        # A - Number of requests that received service
        print(f"A - Number of requests that received service: {self.server.customers_served}")
        
        # B - Number of requests that encountered a full queue and were rejected without receiving service
        print(f"B - Number of requests that were rejected without receiving service: {self.server.customers_rejected}")
        
        # Tend - Time of completion of the last message
        print(f"Tend - Time of completion of the last message: {self.server.last_departure_time}")
        
        # T_w - Average wait time of a message in the server system before receiving service
        print(f"T_w - Average wait time of a message in the server system before receiving service: {self.server.average_wait_time():.4f}")
        
        # T_s - Average service time of a message in the server system
        print(f"T_s - Average service time of a message in the server system: {self.server.average_service_time():.4f}")
    
    def print_average_time_till_departure(self):
        print(f"number of customers served: {self.server.customers_served}")
        print(f"Average time till departure: {(self.server.total_service_time + self.server.total_wait_time) / self.server.customers_served:.4f}")
    
    def run(self, T):
        self.events.put(Event(0, 'ARRIVAL', self.server))
        while not self.events.empty():
            event = self.events.get()
            current_time = event.time
            if current_time > T:
                break
            if event.type == 'ARRIVAL':
                event.server.handle_arrival(current_time)
                next_arrival_time = current_time + random.expovariate(self.lambda_)
                self.events.put(Event(next_arrival_time, 'ARRIVAL', self.server))
            elif event.type == 'DEPARTURE':
                event.server.handle_departure(current_time, event)
        
    

        
        

if __name__ == "__main__":
    # Use current time as seed for different results each run
    seed = int(time.time()) 
    np.random.seed(seed)
    random.seed(seed)
    
    args = sys.argv
    if len(args) != 5: 
        print("wrong number of arguments")
        print("Usage: python simulator.py lambda mu N T")
        sys.exit(1)
    lambda_ = float(args[1])
    mu = float(args[2])
    N = int(args[3])
    T = int(args[4])
    
    # Create and run simulation
    lb = loadBalancer(lambda_, mu, N)
    lb.run(T)
    #lb.print_comprehensive_results()
    lb.print_average_time_till_departure()
    
    