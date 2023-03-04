import numpy as np


def normalize_attributes(data_array, att_types, att_symbol, a, b):
    copy_of_data_array = np.copy(data_array)
    indexes_of_attributes = np.where(att_types[:, 1] == att_symbol)
    numerical_part_of_array = copy_of_data_array[:, indexes_of_attributes].astype(float)
    minimums = np.amin(numerical_part_of_array, axis=0)
    maximums = np.amax(numerical_part_of_array, axis=0)
    numerical_part_of_array = (numerical_part_of_array - minimums) * (b - a) / (maximums - minimums) + a
    copy_of_data_array[:, indexes_of_attributes] = numerical_part_of_array.astype(str)
    return copy_of_data_array


def my_mean(vector):
    return np.sum(vector) / vector.size


def my_variance(vector):
    vector -= my_mean(vector)
    vector *= vector
    return np.sqrt(np.sum(vector) / vector.size)


def standardize_attribute(vector):
    return (vector - my_mean(vector)) / my_variance(vector)


data_from_file = np.loadtxt('cwiczenie01\\dane\\heartdisease.txt', dtype=str)
attribute_types = np.loadtxt('cwiczenie01\\dane\\heartdisease-type.txt', dtype=str)
numeric_attribute = 'n'
symbolic_attribute = 's'
save_some_arrays_to_files = True

separation_line = '\n' * 2 + '#' * 100


# CWICZENIA 01
print('\n\n')
print('################')
print('# CWICZENIA 01 #')
print('################')

# Zadanie 3) a)
print(separation_line)
print('Zadanie 3) a)')
print('Decision classes:\n', np.unique(data_from_file[:, -1]))


# Zadanie 3) b)
print(separation_line)
print('Zadanie 3) b)')
print('Sizes of decision classes.')
decision_classes, size_of_decision_classes = np.unique(data_from_file[:, -1], return_counts=True)
format_string = '{:10}  {:6}'
print(format_string.format('class:', 'size:'))
for index in np.arange(decision_classes.size):
    print(format_string.format(decision_classes[index], size_of_decision_classes[index]))


# Zadanie 3) c)
print(separation_line)
print('Zadanie 3) c)')
indexes_of_num_att = np.where(attribute_types == numeric_attribute)[0]
symbols_of_num_att = attribute_types[indexes_of_num_att, 0]
data_of_num_att = data_from_file[:, indexes_of_num_att].astype(float)
minimums_of_num_att = np.amin(data_of_num_att, axis=0)
maximums_of_num_att = np.amax(data_of_num_att, axis=0)
print("Minimum and maximum values of numeric attributes.")
format_string = '{:12}  {:8}  {:8}'
print(format_string.format('attribute:', 'min:', 'max:'))
for index in np.arange(0, symbols_of_num_att.size):
    print(format_string.format(symbols_of_num_att[index], minimums_of_num_att[index], maximums_of_num_att[index]))


# Zadanie 3) d)
print(separation_line)
print('Zadanie 3) d)')
print('Number of unique values for all attributes.')
format_string = '{:12}  {:20}'
print(format_string.format('attribute:', 'number of values:'))
for index in np.arange(0, attribute_types[:, 0].size):
    print(format_string.format(attribute_types[index, 0], np.unique(data_from_file[:, index]).size))


# Zadanie 3) e)
print(separation_line)
print('Zadanie 3) e)')
format_string = "Set of different available values for attribute '{}':\n{}\n"
for index in np.arange(0, attribute_types[:, 0].size):
    print(format_string.format(attribute_types[index, 0], np.unique(data_from_file[:, index])))


# Zadanie 3) f)
print(separation_line)
print('Zadanie 3) f)')
indexes_of_num_att = np.where(attribute_types == numeric_attribute)[0]
symbols_of_num_att = attribute_types[indexes_of_num_att, 0]
data_of_num_att = data_from_file[:, indexes_of_num_att].astype(float)
standard_deviations_of_num_att = np.std(data_of_num_att, axis=0)
print("Standard deviation of numeric attributes.")
format_string = '{:12}  {:20}'
print(format_string.format('attribute:', 'standard deviation:'))
format_string = '{:12}  {:20.12f}'
for index in np.arange(0, symbols_of_num_att.size):
    print(format_string.format(symbols_of_num_att[index], standard_deviations_of_num_att[index]))


# Zadanie 4) a)
print(separation_line)
print('Zadanie 4) a)')
unknown_value = '?'
generated_data_percentage = 10
preprocessed_data = np.copy(data_from_file)
number_of_objects = preprocessed_data[:, 0].size
number_of_attributes = attribute_types[:, 0].size
for x in np.arange(0, preprocessed_data[:, :-1].size * generated_data_percentage / 100):
    while True:
        i = np.random.randint(number_of_objects)
        j = np.random.randint(number_of_attributes)
        if preprocessed_data[i, j] != unknown_value:
            preprocessed_data[i, j] = unknown_value
            break
print('Number of all attribute values: ', preprocessed_data[:, :-1].size)
print('Number of random unknown attribute values: ', preprocessed_data[preprocessed_data == unknown_value].size)
generated_data = np.copy(preprocessed_data)
for index in np.arange(0, number_of_attributes):
    known_data = generated_data[generated_data[:, index] != unknown_value, index]
    if attribute_types[index, 1] == numeric_attribute:
        mean_value = np.mean(known_data.astype(float))
        generated_data[generated_data[:, index] == unknown_value, index] = mean_value.astype(str)
    elif attribute_types[index, 1] == symbolic_attribute:
        unique_values, counts = np.unique(known_data, return_counts=True)
        most_common_value = unique_values[counts == np.amax(counts)]
        if most_common_value.size > 1:
            most_common_value = np.copy(most_common_value[np.random.randint(most_common_value.size)])
        generated_data[generated_data[:, index] == unknown_value, index] = most_common_value
print('Number of unknown attribute values after data generation: ',
      generated_data[generated_data == unknown_value].size)
if save_some_arrays_to_files:
    np.savetxt('cwiczenie01\\zad4a_missing_data.txt', preprocessed_data, fmt='%s', delimiter=' ', newline='\n')
    np.savetxt('cwiczenie01\\zad4a_generated_data.txt', generated_data, fmt='%s', delimiter=' ', newline='\n')
    print('\nTwo files with full arrays from this exercise have just been saved.')


# Zadanie 4) b)
print(separation_line)
print('Zadanie 4) b)')
indexes_of_num_att = np.where(attribute_types == numeric_attribute)[0]
print('Numeric attributes: ', attribute_types[indexes_of_num_att, 0])

normalized_data_1_1 = normalize_attributes(data_from_file, attribute_types, numeric_attribute, -1, 1)
print('Minimums of normalized data into interval <-1, 1>: ',
      np.amin(normalized_data_1_1[:, indexes_of_num_att].astype(float), axis=0))
print('Maximums of normalized data into interval <-1, 1>: ',
      np.amax(normalized_data_1_1[:, indexes_of_num_att].astype(float), axis=0))

normalized_data_0_1 = normalize_attributes(np.copy(data_from_file), attribute_types, numeric_attribute, 0, 1)
print('Minimums of normalized data into interval <0, 1>: ',
      np.amin(normalized_data_0_1[:, indexes_of_num_att].astype(float), axis=0))
print('Maximums of normalized data into interval <0, 1>: ',
      np.amax(normalized_data_0_1[:, indexes_of_num_att].astype(float), axis=0))

normalized_data_10_10 = normalize_attributes(np.copy(data_from_file), attribute_types, numeric_attribute, -10, 10)
print('Minimums of normalized data into interval <-10, 10>: ',
      np.amin(normalized_data_10_10[:, indexes_of_num_att].astype(float), axis=0))
print('Maximums of normalized data into interval <-10, 10>: ',
      np.amax(normalized_data_10_10[:, indexes_of_num_att].astype(float), axis=0))

print('Minimums of original data: ', np.amin(data_from_file[:, indexes_of_num_att].astype(float), axis=0))
print('Maximums of original data: ', np.amax(data_from_file[:, indexes_of_num_att].astype(float), axis=0))


# Zadanie 4) c)
print(separation_line)
print('Zadanie 4) c)')
standardized_data = np.copy(data_from_file)
indexes_of_num_att = np.where(attribute_types == numeric_attribute)[0]
for index in indexes_of_num_att:
    standardized_data[:, index] = standardize_attribute(standardized_data[:, index].astype(float)).astype(str)

print('Original numeric data.')
format_string = '{:12}  {:12}  {:12}'
print(format_string.format('attribute:', 'mean:', 'standard deviation:'))
format_string = '{:12}  {:12f}  {:20f}'
for index in indexes_of_num_att:
    print(format_string.format(attribute_types[index, 0],
                               my_mean(data_from_file[:, index].astype(float)),
                               my_variance(data_from_file[:, index].astype(float))))
print('\nStandardized numeric data.')
format_string = '{:12}  {:12}  {:12}'
print(format_string.format('attribute:', 'mean:', 'standard deviation:'))
format_string = '{:12}  {:12f}  {:20f}'
for index in indexes_of_num_att:
    print(format_string.format(attribute_types[index, 0],
                               my_mean(standardized_data[:, index].astype(float)),
                               my_variance(standardized_data[:, index].astype(float))))


# Zadanie 4) d)
print(separation_line)
print('Zadanie 4) d)')
data = np.loadtxt('cwiczenie01\\dane\\Churn_Modelling.csv', dtype=str, delimiter=',')
attribute_name = 'Geography'
new_att_value = '1'
filler_att_value = '0'
index_of_att = np.where(data[0, :] == attribute_name)[0][0]
attribute_values = np.unique(data[1:, index_of_att])
modified_data = data[:, 0:index_of_att]
for x in np.arange(0, attribute_values.size):
    column = np.expand_dims(data[:, index_of_att], axis=1)
    modified_data = np.concatenate((modified_data, column), axis=1)
modified_data = np.concatenate((modified_data, data[:, (index_of_att + 1):]), axis=1)
for index in np.arange(0, attribute_values.size):
    column_index = index_of_att + index
    modified_data[1:, column_index] = np.where(modified_data[1:, column_index] == attribute_values[index], '1', '0')
    modified_data[0, column_index] = attribute_values[index]
final_data = np.concatenate((modified_data[:, :index_of_att], modified_data[:, (index_of_att + 1):]), axis=1)

print('Original attributes:\n', data[0, :])
print('\nModified attributes:\n', modified_data[0, :])
print('\nFinal attributes:\n', final_data[0, :])
original_attribute = np.expand_dims(data[:, index_of_att], axis=1)
columns = np.arange(index_of_att, index_of_att + attribute_values.size)
comparison01 = np.concatenate((original_attribute, modified_data[:, columns]), axis=1)
comparison02 = np.concatenate((original_attribute, final_data[:, columns[:-1]]), axis=1)
objects = np.arange(0, 16)
print('\nComparison of original attribute to modified ones for objects from', objects[1], 'to', objects[-1], '\b:')
print(comparison01[objects, :])
print('\nComparison of original attribute to final ones for objects from', objects[1], 'to', objects[-1], '\b:')
print(comparison02[objects, :])
if save_some_arrays_to_files:
    np.savetxt('cwiczenie01\\zad4d_modified_Churn_Modelling.csv', modified_data, fmt='%s', delimiter=',', newline='\n')
    np.savetxt('cwiczenie01\\zad4d_final_Churn_Modelling.csv', final_data, fmt='%s', delimiter=',', newline='\n')
    print('\nTwo files with full arrays from this exercise have just been saved.')


# THE END
print('\n\n###########\n# THE END #\n###########\n')
