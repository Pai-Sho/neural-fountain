#####################################################################
#																	#
#		Iterate through various hyper parameters, storing the		# 
#		distribution of averages for 1000 training sessions of 		#
#		each network topology resulting from the hyper parameters	#
#																	#
#####################################################################

#imports
import fountain

#Constants 				!!!NOTE!!! idk what learning rates to iterate throught. left as constant
N 				  = 10
D_in 			  = 2
D_out 			  = 1
learning_rate 	  = 1e-5
training_sessions = 100

hidden_layer_min_width = 2
hidden_layer_max_width = 4

num_models = (hidden_layer_max_width - hidden_layer_min_width) + (hidden_layer_max_width - hidden_layer_min_width)*(hidden_layer_max_width - hidden_layer_min_width)

#Temporary initializations
depth 			= 0
H_1 			= 0
H_2 			= 0

average_accuracy = 0
stdev 			 = 0
max_accuracy 	 = 0
min_stddev		 = float('inf') 
accuracy_list 	 = []

depth_most_accurate			= depth
H_1_most_accurate			= H_1
H_2_most_accurate			= H_2
N_most_accurate 			= N
D_in_most_accurate 			= D_in
D_out_most_accurate 		= D_out
learning_rate_most_accurate = learning_rate

depth_most_consistent		  = depth
H_1_most_consistent			  = H_1
H_2_most_consistent			  = H_2
N_most_consistent 			  = N
D_in_most_consistent 		  = D_in
D_out_most_consistent 		  = D_out
learning_rate_most_consistent = learning_rate

index = 0

print ""

#Iterate between 1 and 2 hidden layers
for depth in xrange(1,3):

	#For the first hidden layer, test widths from 3 to 10 nodes
	for H_1 in xrange(hidden_layer_min_width,(hidden_layer_max_width+1)):
		
		if(depth == 2):
			#For the second hidden layer, test widths from 3 to 10 nodes
			for H_2 in xrange(hidden_layer_min_width,(hidden_layer_max_width+1)):
				index = index + 1
				#Fancy UI
				print "Training network ", index, " of ",num_models ," with the following hyper parameters for ", training_sessions, " training sessions..."
				print "Input Dimension: ", D_in
				print "Depth:			", depth
				print "Hidden Layer 1 Width:	", H_1
				print "Hidden Layer 2 Width:	", H_2
				print "Output Dimension: ", D_out, "\n"

				for i in xrange (0, training_sessions):
					print "Iteration: ", (i + 1)
					#Train the model and record the accuracy
					accuracy = fountain.TestModel(N, depth, D_in, H_1, H_2, D_out, learning_rate)
					accuracy_list.append(accuracy)

				#end for-i

				average_accuracy = accuracy_list.mean()
				stdev 			 = accuracy_list.stdev()
				print "Average accuracy:	", average_accuracy
				print "Standard deviation:	", stdev, "\n\n"
				#Clear the accuracy list for the next hyper parameter set
				accuracy_list = []

				if(average_accuracy > max_accuracy):
					max_accuracy 				= average_accuracy
					stdev_most_accurate 		= stdev
					depth_most_accurate			= depth
					H_1_most_accurate			= H_1
					H_2_most_accurate			= H_2
					N_most_accurate 			= N
					D_in_most_accurate 			= D_in
					D_out_most_accurate 		= D_out
					learning_rate_most_accurate = learning_rate
				if(stdev < min_stddev):
					accuracy_most_consistent	  = accuracy
					min_stddev					  = stdev
					depth_most_consistent		  = depth
					H_1_most_consistent			  = H_1
					H_2_most_consistent			  = H_2
					N_most_consistent 			  = N
					D_in_most_consistent 		  = D_in
					D_out_most_consistent 		  = D_out
					learning_rate_most_consistent = learning_rate
			#end for-H_2
		else:
			index = index + 1
			#Fancy UI
			print "Training network ", index, " of ",num_models ," with the following hyper parameters for ", training_sessions, " training sessions..."
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

			average_accuracy = accuracy_list.mean()
			stdev 			 = accuracy_list.stdev()
			print "Average Accuracy:	", average_accuracy
			print "Standard Deviation:	", stdev, "\n\n"
			#Clear the accuracy list for the next hyper parameter set
			accuracy_list = []

			#Record the hyper parameters for best accuracy and lowest standard deviation
			if(average_accuracy > max_accuracy):
				max_accuracy 				= average_accuracy
				stdev_most_accurate 		= stdev
				depth_most_accurate			= depth
				H_1_most_accurate			= H_1
				H_2_most_accurate			= H_2
				N_most_accurate 			= N
				D_in_most_accurate 			= D_in
				D_out_most_accurate 		= D_out
				learning_rate_most_accurate = learning_rate
			if(stdev < min_stddev):
				accuracy_most_consistent	  = accuracy
				min_stddev					  = stdev
				depth_most_consistent		  = depth
				H_1_most_consistent			  = H_1
				H_2_most_consistent			  = H_2
				N_most_consistent 			  = N
				D_in_most_consistent 		  = D_in
				D_out_most_consistent 		  = D_out
				learning_rate_most_consistent = learning_rate
		#end if-else
	#end for-H_1
#end for-depth

# Print the two most optimal networks, their accuracy, stdev, and hyper parameters
print "\n\nMost Accurate Network:"
print "\nAccuracy:	", max_accuracy
print "	Standard Deviation:	", stdev_most_accurate
print "	Input Dimension:	", D_in_most_accurate
print "	Hidden Layer Depth:	", depth_most_accurate
print "	Hidden Layer 1 Width:	", H_1_most_accurate
print "	Hidden Layer 2 Width:	", H_1_most_accurate
print "	Output Dimension:	", D_out_most_accurate
print "	Learning Rate:	", learning_rate_most_accurate

print "\n\nMost Consistent Network:"
print "\nAccuracy:	", accuracy_most_consistent
print "	Standard Deviation:	", min_stddev
print "	Input Dimension:	", D_in_most_consistent
print "	Hidden Layer Depth:	", depth_most_consistent
print "	Hidden Layer 1 Width:	", H_1_most_consistent
print "	Hidden Layer 2 Width:	", H_1_most_consistent
print "	Output Dimension:	", D_out_most_consistent
print "	Learning Rate:	", learning_rate_most_consistent

print ""
