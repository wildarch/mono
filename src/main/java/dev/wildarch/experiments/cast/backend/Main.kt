package dev.wildarch.experiments.cast.backend

import com.oracle.bmc.ConfigFileReader
import com.oracle.bmc.Region
import com.oracle.bmc.auth.ConfigFileAuthenticationDetailsProvider
import com.oracle.bmc.objectstorage.ObjectStorageClient
import com.oracle.bmc.objectstorage.model.CreatePreauthenticatedRequestDetails
import com.oracle.bmc.objectstorage.requests.CreatePreauthenticatedRequestRequest
import com.oracle.bmc.objectstorage.requests.GetNamespaceRequest
import com.oracle.bmc.objectstorage.requests.ListObjectsRequest
import java.time.Instant
import java.util.*

fun main() {
    val configFile = ConfigFileReader.parseDefault()
    val provider = ConfigFileAuthenticationDetailsProvider(configFile)
    val osClient = ObjectStorageClient(provider)
    osClient.setRegion(Region.EU_AMSTERDAM_1)
}

private fun listObjects(osClient: ObjectStorageClient) {
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
}

private fun makePar(osClient: ObjectStorageClient) {
    val namespaceResponse = osClient.getNamespace(GetNamespaceRequest.builder().build())
    val namespaceName: String = namespaceResponse.value
    println("Using namespace: $namespaceName")

    val parResponse = osClient.createPreauthenticatedRequest(CreatePreauthenticatedRequestRequest.builder()
        .namespaceName(namespaceName)
        .bucketName("medialib")
        .createPreauthenticatedRequestDetails(CreatePreauthenticatedRequestDetails.builder()
            .name("test-par")
            .objectName("BigBuckBunny.mp4")
            .accessType(CreatePreauthenticatedRequestDetails.AccessType.ObjectRead)
            .timeExpires(Date.from(Instant.now().plusSeconds(600)))
            .build())
        .build())

    val uri = parResponse.preauthenticatedRequest.accessUri

    val fullUri = osClient.endpoint + uri
    print(fullUri)
}