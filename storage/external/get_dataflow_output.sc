import io.joern.dataflowengineoss.passes.reachingdef.ReachingDefPass
import io.joern.dataflowengineoss.passes.reachingdef.ReachingDefProblem
import io.joern.dataflowengineoss.passes.reachingdef.ReachingDefTransferFunction
import io.joern.dataflowengineoss.passes.reachingdef.ReachingDefFlowGraph
import scala.collection.immutable.HashMap
import better.files.File

// https://stackoverflow.com/a/55967256/8999671
def toJson(query: Any): String = query match {
   case m: Map[String, Any] => s"{${m.map(toJson(_)).mkString(",")}}"
   case t: (String, Any) => s""""${t._1}":${toJson(t._2)}"""
   case ss: Seq[Any] => s"""[${ss.map(toJson(_)).mkString(",")}]"""
   case s: String => s""""$s""""
   case null => "null"
   case _ => query.toString
}

/**
  * Instantiate reaching definition problem and print the solution
  *
  * Run with:
  * joern --script storage/external/get_dataflow_output.scala --params filename=x42/c/X42.c,problem=reachingdef
  */
@main def exec(filename: String, cache: Boolean = true) = {
    val summaryFile = File(s"${filename}.dataflow.summary.json")
    if (!summaryFile.exists || !cache) {
        try {
            val cpgFile = File(filename + ".cpg.bin")
            if (cpgFile.exists) {
                println(s"Loading CPG from $cpgFile")
                importCpg(cpgFile.toString)
            }
            else {
                println(s"Exporting CPG to $cpgFile")
                importCode.c(filename)
                run.ossdataflow
                save
                File(project.path + "/cpg.bin").copyTo(cpgFile)
            }

            val nodeFile = File(filename + ".nodes.json")
            val edgeFile = File(filename + ".edges.json")
            if (!nodeFile.exists || !cache) {
                cpg.graph.V.map(node=>node).toJson |> nodeFile.toString
                cpg.graph.E.map(node=>List(node.inNode.id, node.outNode.id, node.label, node.propertiesMap.get("VARIABLE"))).toJson |> edgeFile.toString
            }

            val methods = cpg.method.filter(m => m.filename != "<empty>" && m.name != "<global>").l
            methods.foreach(m => {
                val methodDataflowFile = File(s"${filename}.dataflow.${m.name}.json")
                if (!methodDataflowFile.exists || !cache) {
                    val problem = ReachingDefProblem.create(m)
                    val transferFunction = problem.transferFunction.asInstanceOf[ReachingDefTransferFunction]
                    val numberToNode = problem.flowGraph.asInstanceOf[ReachingDefFlowGraph].numberToNode
                    val df = HashMap(
                        "gen" -> transferFunction.gen.map(kv => (kv._1.id.toString, kv._2.toList.sorted.map(numberToNode).map(_.id).filter(transferFunction.gen.keySet.map(_.id).contains))).toSeq.sortBy(_._1).toMap,
                        "kill" -> transferFunction.kill.map(kv => (kv._1.id.toString, kv._2.toList.sorted.map(numberToNode).map(_.id).filter(transferFunction.gen.keySet.map(_.id).contains))).toSeq.sortBy(_._1).toMap
                    )
                    toJson(df) |> methodDataflowFile.toString
                }
            })
            toJson(methods.map(_.name)) |> s"${filename}.dataflow.summary.json"
            println("Done exporting dataflow")
        }
        catch {
            case e : Exception => {
                println("Error during execution:")
                e.printStackTrace()
            }
        }
        finally {
            try {
                delete
            }
            catch {
                case e : RuntimeException => println(s"Error deleting project: ${e.getMessage}")
            }
        }
    }
    else {
        println(f"result is cached $filename")
    }
}
