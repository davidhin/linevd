import scala.collection.mutable.HashSet

@main def exec(projectName: String, rootType: String) = {
    open(filename)
   
    // Get declaration of typename, resolving aliases
    def trueTypeDecl(tn: String) = {
        val aliasIt = cpg.typeDecl.name(tn).aliasTypeFullName.dedup;
        if (aliasIt.hasNext) {
            cpg.typeDecl.name(aliasIt.head)
        }
        else {
            cpg.typeDecl.name(tn)
        }
    }

    // map TypeDecl to its most grandchild leaf types.
    // Leaf types are external types or internal types without members.
    def mapToMemberTypes(allT: List[TypeDecl], seen: HashSet[String] = HashSet()): List[String] = {
        seen ++= allT.name;
        var toReturn = allT.external.name.l;
        val t = allT.filter(!_.isExternal).l;
        if (!t.member.hasNext) {
            println(s"terminal type ${t.name.l}\n\tseen: $seen")
            seen ++= memberTypeNames;
            toReturn = toReturn ::: t.name.l;
        }
        else {
            val memberTypeNames = t.member.typeFullName.filterNot(seen).l;
            println(s"nonterminal type ${t.name.head}\n\tunseen members: ${memberTypeNames.dedup.l}\n\tseen: $seen")
            seen ++= memberTypeNames;
            toReturn = toReturn ::: memberTypeNames
                .map(m => trueTypeDecl(m).l)
                .flatMap(mapToMemberTypes(_: List[TypeDecl], seen)).dedup.l;
        }
        toReturn
    }
    println(trueTypeDecl(rootType).l)
    println(mapToMemberTypes(trueTypeDecl(rootType).l))
}
