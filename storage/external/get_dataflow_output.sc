import io.joern.dataflowengineoss.passes.reachingdef.ReachingDefPass
import io.joern.dataflowengineoss.passes.reachingdef.ReachingDefProblem
import io.joern.dataflowengineoss.passes.reachingdef.ReachingDefTransferFunction
import io.joern.dataflowengineoss.passes.reachingdef.DataFlowSolver

def runReachingDef() = {
    println(s"Methods: ${cpg.method.l.length}")
    cpg.method.name.foreach { println }
    val mainMethod = cpg.method("main").head

    val pass = new ReachingDefPass(cpg)
    val result = pass.createAndApply()
    println(result)

    val problem = ReachingDefProblem.create(mainMethod)
    val solution = new DataFlowSolver().calculateMopSolutionForwards(problem)
    println(s"Ran reaching def pass.")
    println(s"IN: ${solution.in}")
    println(s"OUT: ${solution.out}")
    val transferFunction = solution.problem.transferFunction.asInstanceOf[ReachingDefTransferFunction]
    println(s"GEN: ${transferFunction.gen}")
    println(s"KILL: ${transferFunction.kill}")

    // TODO: get AST subtree of all definitions in the graph
}

/**
  * Instantiate reaching definition problem and print the solution
  *
  * Run with:
  * joern --script storage/external/get_dataflow_output.scala --params filename=x42/c/X42.c,problem=reachingdef
  */
@main def exec(filename: String, problem: String) = {
    importCode.c(filename)
    run.ossdataflow

    // TODO: run dataflow solver and output to file
    println(s"Run $problem on $filename")
    if (problem == "reachingdef") {
        runReachingDef()
    }

    delete
}
