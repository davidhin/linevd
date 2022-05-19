import better.files.File

@main def exec(filename: String, exportJson: Boolean, exportCpg: Boolean) = {
   importCode(filename)
   run.ossdataflow
   if (exportCpg) {
      save
      File(project.path + "/cpg.bin").copyTo(File(filename + ".cpg.bin"), overwrite=true)
   }
   if (exportJson) {
      cpg.graph.E.map(node=>List(node.inNode.id, node.outNode.id, node.label, node.propertiesMap.get("VARIABLE"))).toJson |> filename + ".edges.json"
      cpg.graph.V.map(node=>node).toJson |> filename + ".nodes.json"
   }
   delete
}

