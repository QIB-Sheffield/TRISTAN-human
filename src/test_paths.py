import os

rel_path = os.path.dirname(__file__)
data_path_file = os.path.join(rel_path, '..', 'data', 'data_paths.txt')
with open(data_path_file, 'r') as file:
    main_data_path_1 = file.readline().strip()
    main_data_path_2 = file.readline().strip()

print(main_data_path_1)
print(main_data_path_2)