import better.files.File

/**
  * Instantiate reaching definition problem and print the solution
  *
  * Run with:
  * joern --script storage/external/get_dataflow_output.scala --params filename=x42/c/X42.c
  */
@main def exec(filename: String, runOssDataflow: Boolean = true, exportJson: Boolean = true, exportCpg: Boolean = true, deleteAfter: Boolean = true) = {
   val cpgFile = File(filename + ".cpg.bin")
   if (cpgFile.exists) {
      println(s"Loading CPG from $cpgFile")
      importCpg(cpgFile.toString)
   }
   else {
      println(s"Exporting CPG to $cpgFile")
      importCode(filename)
      if (runOssDataflow) {
         run.ossdataflow
      }
   }
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
   if (deleteAfter) {
      delete
   }
}

