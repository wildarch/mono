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
                        link(rel = "stylesheet", href = "https://cdn.simplecss.org/simple.min.css")
                    }

                    body {
                        h2 {
                            +"Media"
                        }
                        ul {
                            for (obj in objectsResponse.listObjects.objects) {
                                if (obj.name.endsWith(".vtt")) {
                                    // Skip subtitles
                                    continue
                                }
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
                val subtitlesName = replaceExtension(objectName, "en.vtt")
                val subtitlesUri = try {
                    makePublicObjectUri(osClient, namespaceName, bucketName, subtitlesName)
                } catch (e: Exception) {
                    println("Cannot make uri for subtitles: $e")
                    null
                }
                val videoType = when (objectName.substringAfterLast(".")) {
                    "webm" -> "video/webm"
                    "mp4" -> "video/mp4"
                    else -> null
                }
                call.respondHtml {
                    head {
                        title {
                            +"Medialib - playing '$objectName'"
                        }
                        link(rel = "stylesheet", href = "https://cdn.simplecss.org/simple.min.css")
                    }

                    body {
                        h2 {
                            +"Now playing: '$objectName'"
                        }

                        video {
                            controls = true
                            autoPlay = true
                            attributes["crossorigin"] = "anonymous"

                            source {
                                src = uri
                                if (videoType != null) {
                                    type = videoType
                                }
                            }

                            if (subtitlesUri != null) {
                                track {
                                    attributes["label"] = "English"
                                    attributes["kind"] = "subtitles"
                                    attributes["srclang"] = "en"
                                    attributes["src"] = subtitlesUri
                                    attributes["default"] = ""
                                }
                            }

                            +"Browser does not support the video tag"
                        }

                        button {
                            id = "cast"
                            +"Cast"
                        }

                        script(src = "https://cdnjs.cloudflare.com/ajax/libs/castjs/5.2.0/cast.min.js") {}
                        if (subtitlesUri != null) {
                            script {
                                unsafe {
                                    +"""
                                    const cjs = new Castjs();
                                    document.getElementById('cast').addEventListener('click', function() {
                                        if (cjs.available) {
                                            cjs.cast('${uri}', {
                                                subtitles: [{
                                                    active: true,
                                                    label: 'English',
                                                    src: '${subtitlesUri}'
                                                }],
                                            });
                                        }
                                    });
                                """.trimIndent()
                                }
                            }
                        } else {
                            script {
                                unsafe {
                                    +"""
                                    const cjs = new Castjs();
                                    document.getElementById('cast').addEventListener('click', function() {
                                        if (cjs.available) {
                                            cjs.cast('${uri}');
                                        }
                                    });
                                """.trimIndent()
                                }
                            }
                        }
                    }
                }
            }
        }
    }.start(wait = true)
}

private fun replaceExtension(objectName: String, extension: String): String {
    return objectName.substringBeforeLast('.') + '.' + extension
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

class TRACK(consumer: TagConsumer<*>) :
    HTMLTag("track", consumer, emptyMap(),
        inlineTag = true,
        emptyTag = false), HtmlInlineTag {
}
fun VIDEO.track(block: TRACK.() -> Unit = {}) {
    TRACK(consumer).visit(block)
}