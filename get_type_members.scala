
   //def getAliasedType(typeDecl: io.shiftleft.codepropertygraph.generated.nodes.TypeDecl): TypeDecl = {
   //   if (typeDecl.aliasTypeFullName.isDefined) {
   //      cpg.typeDecl.name(typeDecl.aliasTypeFullName.head).head
   //   }
   //   else {
   //      typeDecl.head
   //   }
   //}
   //def getMembers(typeName: String) = {
   //   cpg.typeDecl.name(typeName).member
   //}


import scala.collection.mutable._

@main def exec(project: String, rootType: String) = {
   // TODO: tell if type is alias

   val rootType = "zval"
   val bannedTypeNames = Set("struct")
   var q = Queue(rootType)
   var hashSet = new HashSet[String]
   var aliasedTypes = new HashSet[String]
   var typeName = rootType
   while (q.nonEmpty) {
      typeName = q.head
      q = q.tail
      if (!hashSet(typeName)) {
         println(s"typeName=$typeName")
         hashSet += typeName
         val trueType = (
            if (cpg.typeDecl.name(typeName).aliasTypeFullName.isEmpty)
               cpg.typeDecl.name(typeName)
            else
               cpg.typeDecl.name(cpg.typeDecl.name(typeName).aliasTypeFullName.head)
         )
         aliasedTypes += trueType.fullName.head
         val uniqueMemberTypes = trueType.member.typeFullName.toSet.diff(bannedTypeNames)
         q ++= uniqueMemberTypes
         println(s"trueType=${trueType.name} uniqueMemberTypes=$uniqueMemberTypes q=$q")
      }
   }
}

