import com.azure.identity.ClientSecretCredentialBuilder
import com.azure.storage.blob.BlobClientBuilder
import java.io.File

object AzureBlobUploader {
  def main(args: Array[String]): Unit = {
    // Azure Data Lake Storage configurations
    val tenantId = "your_tenant_id"           // Azure Tenant ID
    val clientId = "your_client_id"           // Azure Client ID
    val clientSecret = "your_client_secret"   // Azure Client Secret
    val containerName = "your_container_name" // Container name
    val newBlobName = "SAMT_OPT.csv"          // Name for the file in ADLS Gen2

    // Path to the existing local file
    val existingFilePath = "/path/to/your/existing/file.csv"

    // Rename the file locally
    val existingFile = new File(existingFilePath)
    if (!existingFile.exists()) {
      throw new RuntimeException(s"File does not exist: $existingFilePath")
    }

    val renamedFilePath = existingFile.getParent + File.separator + "SAMT_OPT.csv"
    val renamedFile = new File(renamedFilePath)
    if (existingFile.renameTo(renamedFile)) {
      println(s"File renamed to: $renamedFilePath")
    } else {
      throw new RuntimeException("Failed to rename the file.")
    }

    // Step 1: Build ClientSecretCredential
    val clientSecretCredential = new ClientSecretCredentialBuilder()
      .clientId(clientId)
      .clientSecret(clientSecret)
      .tenantId(tenantId)
      .build()

    // Step 2: Construct the BlobClient
    val blobClient = new BlobClientBuilder()
      .credential(clientSecretCredential)
      .endpoint(s"https://your_account_name.blob.core.windows.net") // Replace with your storage account
      .containerName(containerName)
      .blobName(newBlobName)
      .buildClient()

    // Step 3: Upload the renamed file
    blobClient.uploadFromFile(renamedFilePath, true)
    println(s"File uploaded successfully to blob: $newBlobName in container: $containerName")
  }
}
