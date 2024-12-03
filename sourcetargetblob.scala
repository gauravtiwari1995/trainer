import com.azure.storage.blob._
import com.azure.storage.blob.models._

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

object ReadAndUploadADLS {
  def main(args: Array[String]): Unit = {
    // ADLS Gen2 Connection Details
    val accountName = "<your-account-name>"
    val containerName = "<your-container-name>"
    val sourceFilePath = "<source-folder-path>/source-file.csv" // Path in ADLS
    val targetFolderPath = "<target-folder-path>" // Target folder in ADLS
    val targetFileName = "SAMT_OPT.csv" // File name to upload

    // Build the BlobServiceClient using a Client Secret or Managed Identity
    val endpoint = s"https://$accountName.blob.core.windows.net"
    val clientSecretCredential = new com.azure.identity.ClientSecretCredentialBuilder()
      .clientId("<your-client-id>")
      .clientSecret("<your-client-secret>")
      .tenantId("<your-tenant-id>")
      .build()

    val blobServiceClient = new BlobServiceClientBuilder()
      .endpoint(endpoint)
      .credential(clientSecretCredential)
      .buildClient()

    // Get a reference to the container
    val blobContainerClient = blobServiceClient.getBlobContainerClient(containerName)

    // Step 1: Read the source file from ADLS
    val sourceBlobClient = blobContainerClient.getBlobClient(sourceFilePath)
    val blobInputStream = sourceBlobClient.openInputStream()

    // Copy data from BlobInputStream to ByteArrayOutputStream
    val outputStream = new ByteArrayOutputStream()
    val buffer = new Arrayfer
    var bytesRead = blobInputStream.read(buffer)
    while (bytesRead != -1) {
      outputStream.write(buffer, 0, bytesRead)
      bytesRead = blobInputStream.read(buffer)
    }
    blobInputStream.close()

    val fileContent = outputStream.toByteArray
    outputStream.close()

    println(s"Read file from ADLS: $sourceFilePath")

    // Step 2: Upload the file to the target location
    val targetBlobPath = s"$targetFolderPath/$targetFileName"
    val targetBlobClient = blobContainerClient.getBlobClient(targetBlobPath)

    // Upload the file content to the target location
    val inputStream = new ByteArrayInputStream(fileContent)
    targetBlobClient.upload(inputStream, fileContent.length, true)
    inputStream.close()

    println(s"Uploaded file to ADLS: $targetBlobPath")
  }
}
