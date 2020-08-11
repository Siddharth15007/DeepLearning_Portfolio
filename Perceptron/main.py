# And Perceptron for Practice

import pandas as pd

weight1 = 0.0
weight2 = 0.0
bias = 0.0

#input
test_input = [(0,0),(0,1),(1,0),(1,1)]
#output
correct_output = [False,False,False,True]
outputs = []

#Generate Model
for test_input, correct_output in zip(test_input,correct_output):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = int(linear_combination >= 0)
    is_correct_str = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_str])

#print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
print(num_wrong)

output_frame = pd.DataFrame(outputs, columns=['Input 1', ' Input 2', ' Linear Combination', ' Activation Output', ' Is Correct'])

if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))