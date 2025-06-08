#include "read_mat_files.h"

void read_val_col_rowptrs_from_txts(std::vector<PRECISION>& values, std::vector<int>& columns, std::vector<int>& row_ptrs, std::string data_name) {

    //////////////////////////////////////////////////

    std::string line;

    // Open the file
    values.clear();
    columns.clear();
    row_ptrs.clear();
    std::ifstream file("data/values_" + data_name + ".txt");

    // Check if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file." << std::endl;
    }


    while (std::getline(file, line)) {
        try {
            // Convert the string to a PRECISION
            PRECISION value = std::stof(line);
            values.push_back(value);
            // Process the PRECISION value (e.g., print it)
            // std::cout << "Converted value: " << value << std::endl;
        }
        catch (const std::invalid_argument& e) {
            // Handle cases where the string is not a valid number
            std::cerr << "Invalid input: '" << line << "' is not a number." << std::endl;
        }
        catch (const std::out_of_range& e) {
            // Handle cases where the number is out of range for a PRECISION
            std::cerr << "Out of range: '" << line << "' is too large or small for a PRECISION." << std::endl;
        }
    }

    // Close the file
    file.close();

    // READ COLUMN INDEX VALUES FROM TXT FILE
    file.open("data/col_inds_" + data_name + ".txt", std::ifstream::in);

    // Check if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file." << std::endl;
    }

    while (std::getline(file, line)) {
        try {
            // Convert the string to a int
            int column = std::stoi(line);
            columns.push_back(column);
            // Process the double value (e.g., print it)
            // std::cout << "Converted value: " << column << std::endl;
        }
        catch (const std::invalid_argument& e) {
            // Handle cases where the string is not a valid number
            std::cerr << "Invalid input: '" << line << "' is not a number." << std::endl;
        }
        catch (const std::out_of_range& e) {
            // Handle cases where the number is out of range for a double
            std::cerr << "Out of range: '" << line << "' is too large or small for a double." << std::endl;
        }
    }

    // Close the file
    file.close();

    // READ ROW POINTERS FROM TXT FILE
    file.open("data/row_ptrs_" + data_name + ".txt", std::ifstream::in);

    // Check if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file." << std::endl;
    }

    while (std::getline(file, line)) {
        try {
            // Convert the string to a int
            int row_ptr = std::stoi(line);
            row_ptrs.push_back(row_ptr);
            // Process the double value (e.g., print it)
            // std::cout << "Converted value: " << row_ptr << std::endl;
        }
        catch (const std::invalid_argument& e) {
            // Handle cases where the string is not a valid number
            std::cerr << "Invalid input: '" << line << "' is not a number." << std::endl;
        }
        catch (const std::out_of_range& e) {
            // Handle cases where the number is out of range for a double
            std::cerr << "Out of range: '" << line << "' is too large or small for a double." << std::endl;
        }
    }

    // Close the file
    file.close();


    ///////////////////////////////////////////////////////////////////////
}

void read_b_from_txt(std::vector<PRECISION>& b, std::string data_name) {

    //////////////////////////////////////////////////

    std::string line;

    // Open the file
    b.clear();
    std::ifstream file("data/b_" + data_name + ".txt");

    // Check if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file." << std::endl;
    }


    while (std::getline(file, line)) {
        try {
            // Convert the string to a PRECISION
            PRECISION value = std::stof(line);
            b.push_back(value);
            // Process the PRECISION value (e.g., print it)
            // std::cout << "Converted value: " << value << std::endl;
        }
        catch (const std::invalid_argument& e) {
            // Handle cases where the string is not a valid number
            std::cerr << "Invalid input: '" << line << "' is not a number." << std::endl;
        }
        catch (const std::out_of_range& e) {
            // Handle cases where the number is out of range for a PRECISION
            std::cerr << "Out of range: '" << line << "' is too large or small for a PRECISION." << std::endl;
        }
    }

    // Close the file
    file.close();


    ///////////////////////////////////////////////////////////////////////
}
