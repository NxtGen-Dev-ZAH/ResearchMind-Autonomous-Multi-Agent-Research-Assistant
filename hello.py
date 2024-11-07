import os
import langchain


def main():
    print("Hello from agentai!")

    # Get the current directory
    current_dir = os.getcwd()

    # List all files in the current directory
    files = os.listdir(current_dir)

    # Print the file names
    print("Files in the current directory:")
    for file in files:
        print(file)

    # Print the total number of files
    print(f"Total number of files: {len(files)}")
    print("I am fucking great")

if __name__ == "__main__":
    main()
