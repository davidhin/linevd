# %%
from cpgqls_client import CPGQLSClient, import_code_query, workspace_query

server_endpoint = "localhost:8088"
# basic_auth_credentials = ("username", "password")
# client = CPGQLSClient(server_endpoint, auth_credentials=basic_auth_credentials)
client = CPGQLSClient(server_endpoint)

# %%
print("execute a simple CPGQuery")
query = "val a = 1"
result = client.execute(query)
print(result)

# %%
print("execute a `workspace` CPGQuery")
query = workspace_query()
result = client.execute(query)
print(result)

# %%
print("execute an `importCode` CPGQuery for C")
query = import_code_query("/work/LAS/weile-lab/benjis/weile-lab/linevd/x42/c", "my-c-project")
result = client.execute(query)
print(result)

# %%
# query = """importCode.newc(inputPath="/work/LAS/weile-lab/benjis/weile-lab/linevd/x42/c", projectName="my-c-project", language="C")"""
# result = client.execute(query)

# %%
print("execute an `importCode` CPGQuery for Java")
query = import_code_query("/work/LAS/weile-lab/benjis/weile-lab/linevd/x42/java/X42.jar", "my-java-project")
result = client.execute(query)
print(result)
