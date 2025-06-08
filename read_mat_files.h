#ifndef READ_MAT_FILES_H
#define READ_MAT_FILES_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept> // For std::invalid_argument and std::out_of_range

#define PRECISION float

// Function to read values, columns, and row pointers from text files
void read_val_col_rowptrs_from_txts(std::vector<PRECISION>& values, std::vector<int>& columns, std::vector<int>& row_ptrs, std::string data_name);
void read_b_from_txt(std::vector<PRECISION>& b, std::string data_name);

#endif // READ_MAT_FILES_H