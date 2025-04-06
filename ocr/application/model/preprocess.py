import html
import pandas as pd

def convert_dataset(dataset_dir):
    out_df = pd.DataFrame({"file_name": [], "text": []})
    with open(f"{dataset_dir}/linux_gt.txt", "r") as f:
        lines = f.readlines()
        lines = [[line.split()[0], line.split()[1]] for line in lines]

    previous_file_piece = lines[0][0].split("/")[3]
    previous_file_components = previous_file_piece.split("-")
    previous_file_components.pop(-1)
    previous_file = "-".join(previous_file_components) + ".png"

    resulting_text = ""

    for line in lines:
        file_path = line[0]
        string = line[1]

        current_file_piece = file_path.split("/")[3]
        current_file_components = current_file_piece.split("-")
        current_file_components.pop(-1)
        current_file = "-".join(current_file_components) + ".png"

        if current_file == previous_file:
            if resulting_text != "":
                resulting_text += " "
            resulting_text += html.unescape(str(string))
        else:
            out_df.loc[len(out_df)] = [previous_file, resulting_text]
            previous_file = current_file
            resulting_text = html.unescape(str(string))

    print(out_df)


if __name__ == '__main__':
    convert_dataset("datasets/iam/iam_dataset")