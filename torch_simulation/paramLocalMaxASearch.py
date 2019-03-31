#####################################################################
#                                                                   #
#       Iterate through the hyper parameters batch size and         # 
#       learning rate for a given network topology. Write the       #
#       average accuracy and its standard deviation to a file       #
#                                                                   #
#####################################################################

#imports
import fountain
import statistics

# Initialize file I/O
file = "results/param_local_max_a_search_results"
file_to_write = open(file, 'w+')

# Define network topology (fixed for all training sessions)
D_in                = 2
H_1                 = 3
H_2                 = 3
D_out               = 1
training_sessions   = 10

# Define hyperparameters to test
learning_rates  = [0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
batch_sizes     = [6, 7, 8, 9, 10]

# UI related veriables
total = len(learning_rates)*len(batch_sizes)
count = 1

# Initialize statistics variables
average_accuracy = 0
stdev 			 = 0
max_accuracy 	 = 0
min_stddev		 = float('inf')
accuracy_list    = []

#Formatting
print ""

# Iterate through learning rates
for _,lr in enumerate(learning_rates):

    # Iterate through batch sizes
    for _,N in enumerate(batch_sizes):
        print "Parameter setting ", count, " of ", total
        count = count + 1
        print("Batch Size       = %d \nLearning Rate    = %f\n"%(N,lr))

        # Iterate through training runs
        for i in range(training_sessions):
            accuracy_list.append(fountain.TestModel(N,2,D_in,H_1,H_2,D_out,lr))

        # Calculate, print and save mean and std. dev. of accuracy
        average_accuracy = sum(accuracy_list)/len(accuracy_list)
        stdev 			 = statistics.stdev(accuracy_list)

        print "\nAverage Accuracy:	", average_accuracy
        print "Standard Deviation:	", stdev, "\n\n"

        file_to_write.write("For batch size = %d, lr = %f:\n"%(N,lr))
        file_to_write.write("Average Accuracy:\t\t%f\n"%(average_accuracy))
        file_to_write.write("Standard Deviation:\t\t%f\n\n"%(stdev))

        #Clear the list after you use the values
        accuracy_list = []
    #end for-batch_size
#end for-learning rate
file_to_write.close()