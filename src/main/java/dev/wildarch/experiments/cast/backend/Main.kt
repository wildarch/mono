package dev.wildarch.experiments.cast.backend

import com.oracle.bmc.ConfigFileReader
import com.oracle.bmc.Region
import com.oracle.bmc.auth.ConfigFileAuthenticationDetailsProvider
import com.oracle.bmc.objectstorage.ObjectStorageClient
import com.oracle.bmc.objectstorage.requests.GetNamespaceRequest
import com.oracle.bmc.objectstorage.requests.ListObjectsRequest

fun main() {
    val configFile = ConfigFileReader.parseDefault()
    val provider = ConfigFileAuthenticationDetailsProvider(configFile)
    val osClient = ObjectStorageClient(provider)
    osClient.setRegion(Region.EU_AMSTERDAM_1)

    val namespaceResponse = osClient.getNamespace(GetNamespaceRequest.builder().build())
    val namespaceName: String = namespaceResponse.value
    println("Using namespace: $namespaceName")

    val objectsResponse = osClient.listObjects(ListObjectsRequest.builder()
        .fields("size")
        .namespaceName(namespaceName)
        .bucketName("medialib")
        .build())

    println("Objects:")
    for (obj in objectsResponse.listObjects.objects) {
        println("${obj.name} ${obj.size / 1_000_000.0}M")
    }

    println("Hello, World!")
}
