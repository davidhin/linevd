import scala.collection.mutable.HashSet

// Get declaration of typename, resolving aliases
def trueTypeDecl(tn: String) = {
    val aliasIt = cpg.typeDecl.name(tn).aliasTypeFullName.dedup;
    if (aliasIt.hasNext) {
        val aliasedType = aliasIt.head;
        if (aliasedType.startsWith("anonymous_type_")) {
            val anonNum = aliasedType.substring("anonymous_type_".length).toInt;
            cpg.typeDecl.name("").filename(cpg.typeDecl.name(tn).filename.head).sortBy(_.order).drop(anonNum).take(1)
        }
        else {
            cpg.typeDecl.name(aliasedType)
        }
        
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
        toReturn = toReturn ::: t.name.l;
    }
    else {
        val memberTypeNames = t.member.typeFullName.filterNot(seen).l;
        seen ++= memberTypeNames;
        toReturn = toReturn ::: memberTypeNames
            .map(m => trueTypeDecl(m).l)
            .flatMap(mapToMemberTypes(_: List[TypeDecl], seen)).dedup.l;
    }
    toReturn
}

@main def exec(projectName: String, rootType: String) = {
    open(projectName)
    
    val trueType = trueTypeDecl(rootType).dedup.l;
    mapToMemberTypes(trueType).mkString("\n") |> s"memberTypes_$rootType.txt"
}
