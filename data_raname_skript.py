import os


def remove_total_from_filenames(folder_path):
    """
    Batch remove '_total' from filenames
    Example: '00000_03_total.png' → '00000_03.png'
    """
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return

    # Get all files in the folder
    files = os.listdir(folder_path)

    # Counter
    renamed_count = 0

    for filename in files:
        # Full file path
        old_path = os.path.join(folder_path, filename)

        # Only process files, not directories
        if not os.path.isfile(old_path):
            continue

        # If filename contains '_total'
        if '_total' in filename:
            # Remove '_total'
            new_filename = filename.replace('_total', '')
            new_path = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {new_filename}")
            renamed_count += 1

    print(f"\nCompleted! Renamed {renamed_count} files")


# Usage
if __name__ == "__main__":
    # Replace with your folder path
    folder_path = "/home/tzhang/sythetic_data/images"

    # Call the function
    remove_total_from_filenames(folder_path)