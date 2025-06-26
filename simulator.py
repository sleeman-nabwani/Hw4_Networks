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
    def __init__(self, lambda_, mu, P, Q, M):
        self.lambda_ = lambda_
        self.mu = mu
        self.events = PriorityQueue()
        self.servers = [Server(mu[i], Q[i], self.events) for i in range(M)] 
        self.servers_probabilities = P
        
    def run(self, T):
        #initialize events
        first_server = np.random.choice(self.servers, p=self.servers_probabilities)
        self.events.put(Event(0, 'ARRIVAL', first_server))
        while not self.events.empty():
            event = self.events.get()
            current_time = event.time
            if current_time > T:
                break
            #handle arrival events
            if event.type == 'ARRIVAL':
                event.server.handle_arrival(current_time)
                next_arrival_time = current_time + random.expovariate(self.lambda_)
                #choose server based on probabilities
                chosen_server = np.random.choice(self.servers, p=self.servers_probabilities)
                self.events.put(Event(next_arrival_time, 'ARRIVAL', chosen_server))
            #handle departure events
            elif event.type == 'DEPARTURE':
                event.server.handle_departure(current_time, event)
        
    

    def print_comprehensive_results(self):
        # A - Number of requests that received service
        print(f"A - Number of requests that received service: {sum([server.customers_served for server in self.servers])}")
        
        # B - Number of requests that encountered a full queue and were rejected without receiving service
        print(f"B - Number of requests that were rejected without receiving service: {sum([server.customers_rejected for server in self.servers])}")
        
        # Tend - Time of completion of the last message
        print(f"Tend - Time of completion of the last message: {max([server.last_departure_time for server in self.servers])}")
        
        # T_w - Average wait time of a message in the server system before receiving service
        print(f"T_w - Average wait time of a message in the server system before receiving service: {sum([server.total_wait_time for server in self.servers]) / sum([server.customers_served for server in self.servers]):.4f}")
        
        # T_s - Average service time of a message in the server system
        print(f"T_s - Average service time of a message in the server system: {sum([server.total_service_time for server in self.servers]) / sum([server.customers_served for server in self.servers]):.4f}")
    
    def print_average_time_till_departure(self):
        print(f"number of customers served: {sum([server.customers_served for server in self.servers])}")
        total_wait_time = sum([server.total_wait_time for server in self.servers])
        total_service_time = sum([server.total_service_time for server in self.servers])
        total_customers_served = sum([server.customers_served for server in self.servers])
        print(f"Average time till departure: {(total_wait_time + total_service_time) / total_customers_served:.4f}")
        
    def print_results(self):
        A = sum([server.customers_served for server in self.servers])
        B = sum([server.customers_rejected for server in self.servers])
        T_end = max([server.last_departure_time for server in self.servers])
        T_w = sum([server.total_wait_time for server in self.servers]) / sum([server.customers_served for server in self.servers])
        T_s = sum([server.total_service_time for server in self.servers]) / sum([server.customers_served for server in self.servers])
        print(f"{A} {B} {T_end} {T_w:.4f} {T_s:.4f}")


if __name__ == "__main__":
    # Use current time as seed for different results each run
    seed = int(time.time()) 
    np.random.seed(seed)
    random.seed(seed)
    
    args = sys.argv
    T = int(args[1])
    M = int(args[2])
    if len(args)-1 != (3*M + 3):
        print("wrong number of arguments")
        sys.exit(1)
        
    #parsing to indexes
    P_start = 3
    P_end = M + P_start
    
    lambda_index = P_end
    
    Q_start = lambda_index + 1
    Q_end = Q_start + M
    
    mu_start = Q_end
    mu_end = mu_start + M
    
    #harvesting parameters
    P = [float(args[i]) for i in range(P_start, P_end)]
    lambda_ = float(args[lambda_index])
    Q = [int(args[i]) for i in range(Q_start, Q_end)]
    mu = [float(args[i]) for i in range(mu_start, mu_end)]
    if sum(P) != 1:
        print("sum of P is not 1")
        sys.exit(1)
    
    # Create and run simulation
    lb = loadBalancer(lambda_, mu, P, Q, M)
    lb.run(T)
    lb.print_results()
    
    