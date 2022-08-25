package dev.wildarch.experiments.cast.backend

import com.oracle.bmc.ConfigFileReader
import com.oracle.bmc.Region
import com.oracle.bmc.auth.ConfigFileAuthenticationDetailsProvider
import com.oracle.bmc.objectstorage.ObjectStorageClient
import com.oracle.bmc.objectstorage.model.CreatePreauthenticatedRequestDetails
import com.oracle.bmc.objectstorage.requests.CreatePreauthenticatedRequestRequest
import com.oracle.bmc.objectstorage.requests.GetNamespaceRequest
import com.oracle.bmc.objectstorage.requests.ListObjectsRequest
import io.ktor.server.application.*
import io.ktor.server.html.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.routing.*
import kotlinx.html.*
import io.ktor.http.*
import java.time.Instant
import java.util.*

fun main() {
    val osClient = getObjectStorageClient()
    val namespaceName = getNamespaceName(osClient)
    val bucketName = "medialib"

    val prefix = System.getenv("CAST_BACKEND_PREFIX") ?: ""
    val port = System.getenv("CAST_BACKEND_PORT")?.let { Integer.parseInt(it) } ?: 8080

    embeddedServer(Netty, port = port, host = "127.0.0.1") {
        routing {
            get("/") {
                // List media
                val objectsResponse = osClient.listObjects(
                    ListObjectsRequest.builder()
                        .namespaceName(namespaceName)
                        .bucketName(bucketName)
                        .build()
                )

                call.respondHtml(HttpStatusCode.OK) {
                    head {
                        title {
                            +"Medialib"
                        }
                    }

                    body {
                        h2 {
                            +"Media"
                        }
                        ul {
                            for (obj in objectsResponse.listObjects.objects) {
                                li {
                                    a(href = "${prefix}/play/${obj.name}") {
                                        +obj.name
                                    }
                                }
                            }
                        }
                    }
                }
            }

            get("/play/{object_name}") {
                val objectName = call.parameters["object_name"]!!

                val uri = makePublicObjectUri(osClient, namespaceName, bucketName, objectName)
                call.respondHtml {
                    head {
                        title {
                            +"Medialib - playing '$objectName'"
                        }
                    }

                    body {
                        h2 {
                            +"Now playing: '$objectName'"
                        }

                        video {
                            width = "320"
                            height = "240"
                            controls = true
                            autoPlay = true

                            source {
                                src = uri
                                type = "video/mp4"
                            }

                            +"Browser does not support the video tag"
                        }
                    }
                }
            }
        }
    }.start(wait = true)
}

fun makePublicObjectUri(
    osClient: ObjectStorageClient,
    namespaceName: String,
    bucketName: String,
    objectName: String
): String {
    val parResponse = osClient.createPreauthenticatedRequest(
        CreatePreauthenticatedRequestRequest.builder()
            .namespaceName(namespaceName)
            .bucketName(bucketName)
            .createPreauthenticatedRequestDetails(
                CreatePreauthenticatedRequestDetails.builder()
                    .name("medialib-play")
                    .objectName(objectName)
                    .accessType(CreatePreauthenticatedRequestDetails.AccessType.ObjectRead)
                    .timeExpires(Date.from(Instant.now().plusSeconds(600)))
                    .build()
            )
            .build()
    )

    val uri = parResponse.preauthenticatedRequest.accessUri

    return osClient.endpoint + uri
}

private fun getObjectStorageClient(): ObjectStorageClient {
    val configFile = ConfigFileReader.parseDefault()
    val provider = ConfigFileAuthenticationDetailsProvider(configFile)
    val client = ObjectStorageClient(provider)
    client.setRegion(Region.EU_AMSTERDAM_1)
    return client
}

private fun getNamespaceName(client: ObjectStorageClient): String {
    val namespaceResponse = client.getNamespace(GetNamespaceRequest.builder().build())
    return namespaceResponse.value
}