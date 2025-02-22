# Assignment 2: Social and Large-Scale Networks

## Overview: 
This assignment analyzes graphs stored in Graph Modeling Language (GML) format and:
* Divide the graph into n components.
* Compute clustering coeffients and neighborhood overlap.
* Verfiy homophily and graph balance. 
* Save the updated graph to a file. 

## Requirements: 

install required libraries using: 
 ```sh 
pip install networkx matplotlib numpy scipy
 ```

## How to Run Code: 
Run the script in ther terminal with: 

 ```bash
python ./graph_analysis.py graph_file.gml --components n --plot [C|N|P] --verify_homophily --verify_balanced_graph --output out_graph_file.gml
 ```
[!TIP] 
> Close the graph once done revising it to open a new graph. 
