"""
    Search through hyperparameters (batch size and learning rate) for a given
    network topology
"""

#imports
import fountain
import statistics

# Define network topology (fixed for all training sessions)
D_in = 2
H_1 = 3
H_2 = 3
D_out = 1
training_sessions = 2

# Define hyperparameters to test
learning_rates = [1e-5,5e-5,1e-4,5e-4,1e-3,5e-3]
batch_sizes = [8,16,32,64]

# Initialize statistics variables
average_accuracy = 0
stdev 			 = 0
max_accuracy 	 = 0
min_stddev		 = float('inf')

# Initialize file I/O
file = "results/param_search_results"
file_to_write = open(file, 'w+')

# Iterate through learning rates
for _,lr in enumerate(learning_rates):

    # Iterate through batch sizes
    for _,N in enumerate(batch_sizes):

        print("For batch size = %d, lr = %f:\n"%(N,lr))
        accuracy_list = []

        # Iterate through training runs
        for i in range(training_sessions):

            accuracy_list.append(fountain.TestModel(N,2,D_in,H_1,H_2,D_out,lr))

        # Calculate, print and save mean and std. dev. of accuracy
        average_accuracy = sum(accuracy_list)/len(accuracy_list)
        stdev 			 = statistics.stdev(accuracy_list)

        print("Average Accuracy:	", average_accuracy)
        print("Standard Deviation:	", stdev, "\n\n")

        file_to_write.write("For batch size = %d, lr = %f:\n"%(N,lr))
        file_to_write.write("Average Accuracy:\t\t%f\n"%(average_accuracy))
        file_to_write.write("Standard Deviation:\t\t%f\n\n"%(stdev))

file_to_write.close()
