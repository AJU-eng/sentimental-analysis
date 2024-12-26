from graphviz import Digraph

# Create a new directed graph
dot = Digraph('Server Architecture', format='png')

# Client
dot.node('Client', 'Client (Browser/App)', shape='box', style='filled', color='lightblue')

# PHP Server
dot.node('PHP', 'PHP Server (Main)', shape='box', style='filled', color='lightgreen')

# Python NLP Server
dot.node('Python', 'Python NLP Server', shape='box', style='filled', color='orange')

# MySQL Database
dot.node('MySQL', 'MySQL Database', shape='cylinder', style='filled', color='lightgrey')

# Connections
dot.edge('Client', 'PHP', label='HTTPS Request')
dot.edge('PHP', 'Python', label='REST API/gRPC')
dot.edge('PHP', 'MySQL', label='Database Query')
dot.edge('Python', 'MySQL', label='Database Query/Update')

# Render the graph to a file
dot.render('server_architecture', view=True)

