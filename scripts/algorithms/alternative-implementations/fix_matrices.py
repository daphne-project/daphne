import argparse

def add_ones(input_file, output_file):
    with open(output_file, 'w') as file:
        with open(input_file, 'r') as rfile:
            is_actual_data = False
            is_header = True
            for line in rfile:
                if is_header:
                    file.write(line.replace("pattern", "real"))
                    is_header = False
                if not line.startswith("%"):
                    if not is_actual_data:
                        file.write(line)
                        is_actual_data = True
                    else:
                        r, c = int(line.split()[0]), int(line.split()[1])
                        file.write(f"{r} {c} 1.0\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Explicitly add ones in an MatrixMarket matrix')
    parser.add_argument('--input', type=str, help='Input matrix', required=True)
    parser.add_argument('--output', type=str, help='Path to store the output matrix', required=True)

    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    add_ones(input_file, output_file)
