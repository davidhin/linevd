import better.files.File

@main def exec(filename: String, exportJson: Boolean, exportCpg: Boolean) = {
   importCode(filename)
   run.ossdataflow
   if (exportCpg) {
      save
      val outputFilename = filename + ".cpg.bin"
      println(s"Exporting CPG to $outputFilename")
      File(project.path + "/cpg.bin").copyTo(File(outputFilename), overwrite=true)
   }
   if (exportJson) {
      val nodeOutputFilename = filename + ".nodes.json"
      val edgeOutputFilename = filename + ".edges.json"
      println(s"Exporting JSON to $nodeOutputFilename $edgeOutputFilename")
      cpg.graph.E.map(node=>List(node.inNode.id, node.outNode.id, node.label, node.propertiesMap.get("VARIABLE"))).toJson |> edgeOutputFilename
      cpg.graph.V.map(node=>node).toJson |> nodeOutputFilename
   }
   delete
}

