import os
import re
import requests


def to_snake_case(name):
    name = name.lower()
    name = re.sub(r'[\W\s]+', '_', name)
    return name.strip('_')


def main():
    paper_name = input('Enter the paper name: ')
    paper_link = input('Enter the link to the paper: ')

    folder_name = to_snake_case(paper_name)
    base_path = os.path.join('papers', folder_name)

    # Create folders
    os.makedirs(os.path.join(base_path, 'src'), exist_ok=True)

    # Download paper.pdf
    if paper_link:
        response = requests.get(paper_link)
        if response.status_code == 200:
            with open(os.path.join(base_path, 'paper.pdf'), 'wb') as f:
                f.write(response.content)
            print('Downloaded paper.pdf')
        else:
            print('Failed to download paper.pdf')

    # Create README.md
    readme_content = f"""# {paper_name}

This directory contains an implementation of the paper ["{paper_name}"]({paper_link}).

## Overview

## Repository Structure

- `src/`: Source code for the implementation.
- `paper.pdf`: The original paper.

## Prerequisites

## Dataset Preparation

## Training the Model

## Evaluation

## References

- Original Paper: "[{paper_name}]({paper_link})" can be found [here](paper.pdf).
"""
    with open(os.path.join(base_path, 'README.md'), 'w') as f:
        f.write(readme_content)

    # Create src/__init__.py and src/{name}.py
    open(os.path.join(base_path, 'src', '__init__.py'), 'w').close()
    open(os.path.join(base_path, 'src', f'{folder_name}.py'), 'w').close()

    print(f'Created project structure in {base_path}')


if __name__ == '__main__':
    main()
