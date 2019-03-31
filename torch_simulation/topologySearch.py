#######################################################################
#																	  #
#		Iterate through various hyper parameters related to 		  #
# 		network topology, storing the mean and standard deviation 	  # 
#		of the distribution of averages for some training sessions of #
#		each network topology resulting from the hyper parameters.	  #
#																	  #
#######################################################################

#imports
import fountain
import statistics

# Initialize file I/O
file = "results/hyper_parameter_results"
file_to_write = open(file, 'w')

# Define network topology constants
N 				  = 10
D_in 			  = 2
D_out 			  = 1
learning_rate 	  = 1e-5
training_sessions = 50

# Define network topology variables and ranges
hidden_layer_min_width = 2
hidden_layer_max_width = 4
depth 			= 0
H_1 			= 0
H_2 			= 0

# UI related veriables
total = (hidden_layer_max_width - hidden_layer_min_width + 1) + (hidden_layer_max_width - hidden_layer_min_width + 1)*(hidden_layer_max_width - hidden_layer_min_width + 1)
count = 1

# Initialize statistics variables
average_accuracy = 0
stdev 			 = 0
max_accuracy 	 = 0
min_stddev		 = float('inf') 
accuracy_list 	 = []

#Formatting
print ""

#Iterate between 1 and 2 hidden layers
for depth in xrange(1,3):

	#For the first hidden layer, test widths from 3 to 10 nodes
	for H_1 in xrange(hidden_layer_min_width,(hidden_layer_max_width+1)):
		
		if(depth == 2):
			#For the second hidden layer, test widths from 3 to 10 nodes
			for H_2 in xrange(hidden_layer_min_width,(hidden_layer_max_width+1)):
				#Fancy UI
				print "Training network ", count, " of ",total ," with the following hyper parameters for ", training_sessions, " training sessions..."
				print "Input Dimension: ", D_in
				print "Depth:			", depth
				print "Hidden Layer 1 Width:	", H_1
				print "Hidden Layer 2 Width:	", H_2
				print "Output Dimension: ", D_out, "\n"
				count = count + 1

				for i in xrange (0, training_sessions):
					print "Iteration: ", (i + 1)
					#Train the model and record the accuracy
					accuracy = fountain.TestModel(N, depth, D_in, H_1, H_2, D_out, learning_rate)
					accuracy_list.append(accuracy)

				#end for-i

				average_accuracy = sum(accuracy_list)/len(accuracy_list)
				stdev 			 = statistics.stdev(accuracy_list)
				print "Average Accuracy:	", average_accuracy
				print "Standard Deviation:	", stdev, "\n\n"

				#Write to File
				file_to_write.write( "Input Dimension: 	" + str(D_in) + "\n")
				file_to_write.write( "Hidden Layer 1 Width:	" + str(H_1) + "\n")
				file_to_write.write( "Hidden Layer 2 Width:	" + str(H_2) + "\n")
				file_to_write.write( "Output Dimension: 	" + str(D_out) + "\n")
				file_to_write.write( "Average accuracy:	" + str(average_accuracy) + "\n")
				file_to_write.write( "Standard deviation:	" + str(stdev) + "\n\n")
				#Clear the accuracy list for the next hyper parameter set
				accuracy_list = []
			#end for-H_2
		else:
			count = count + 1
			#Fancy UI
			print "Training network ", count, " of ",total ," with the following hyper parameters for ", training_sessions, " training sessions..."
			print "Input Dimension: ", D_in
			print "Depth:	", depth
			print "Hidden Layer 1 Width:	", H_1
			print "Output Dimension: ", D_out, "\n"

			for i in xrange (0, training_sessions):
				print "Iteration: ", (i + 1)
				#Train the model and record the accuracy
				accuracy = fountain.TestModel(N, depth, D_in, H_1, H_2, D_out, learning_rate)
				accuracy_list.append(accuracy)
			#end for-i

			average_accuracy = sum(accuracy_list)/len(accuracy_list)
			stdev 			 = statistics.stdev(accuracy_list)
			print "\nAverage Accuracy:	", average_accuracy
			print "Standard Deviation:	", stdev, "\n\n"

			#Write to File
			file_to_write.write( "Input Dimension: 	" + str(D_in) + "\n")
			file_to_write.write( "Hidden Layer 1 Width:	" + str(H_1) + "\n")
			file_to_write.write( "Hidden Layer 2 Width:	" + str(H_2) + "\n")
			file_to_write.write( "Output Dimension: 	" + str(D_out) + "\n")
			file_to_write.write( "Average accuracy:	" + str(average_accuracy) + "\n")
			file_to_write.write( "Standard deviation:	" + str(stdev) + "\n\n")
			
			#Clear the accuracy list for the next hyper parameter set
			accuracy_list = []
		#end if-else
	#end for-H_1
#end for-depth
file_to_write.close()