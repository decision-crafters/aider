from setuptools import setup, find_packages

setup(
    name="aider-chat",
    version="0.75.2.dev",
    packages=find_packages(),
    package_data={"aider": ["resources/*", "queries/tree-sitter-language-pack/*", "queries/tree-sitter-languages/*"]},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "aider=aider.__main__:main",
        ],
    },
) 