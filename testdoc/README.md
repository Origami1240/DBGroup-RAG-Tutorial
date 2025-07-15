# AlayaDBLite

AlayaDBLite is a high-performance library for building and querying large-scale approximate nearest neighbor (ANN) graphs. It provides efficient algorithms for constructing and searching ANN graphs, making it suitable for various applications such as recommendation systems, image retrieval, and more.

## Features

- High-performance ANN graph construction
- Efficient querying of ANN graphs
- Support for multiple distance metrics
- Multi-threaded implementation for faster processing

## Installation

### Prerequisites

- C++17 or later
- CMake 3.10 or later
- A compatible C++ compiler (e.g., GCC, Clang)

### Building from Source

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/AlayaDBLite.git
    cd AlayaDBLite
    ```

2. Create a build directory and navigate to it:

    ```sh
    mkdir build
    cd build
    ```

3. Configure the project using CMake:

    ```sh
    cmake ..
    ```

4. Build the project:

    ```sh
    make -j$(nproc)
    ```

## Usage

AlayaDBLite provides a simple interface for building and querying ANN graphs. More detailed usage instructions will be provided in the [documentation](pyalaya/README.md) .

## Testing

To run the tests for AlayaDBLite, follow these steps:

1. Ensure you have built the project as described in the Installation section.

2. Navigate to the build directory:

    ```sh
    cd build/tests
    ```

3. Run the tests :

    ```sh
    ./test_name
    ```

This will execute all the tests and display the results, including any failures.

### Running Specific Tests

You can use the `--gtest_filter` parameter to run specific tests. For example, to run only the tests in the `MyTestSuite` test suite, use the following command:

```sh
./test_name --gtest_filter=MyTestSuite.*
```

## Contributing

We welcome contributions to AlayaDBLite! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with a descriptive message.
4. Push your changes to your forked repository.
5. Create a pull request to the main repository.

Please ensure that your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

We would like to thank all the contributors and users of AlayaDBLite for their support and feedback.

## Contact

If you have any questions or suggestions, please feel free to contact us at [AlayaDBLite@example.com].