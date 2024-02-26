package io.github.anantharajuc.nlpservice;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

@RestController
public class NLPController {

    // Update these variables
    private String uploadPath = "/home/anantha/ARC/AIML_NLP/audio_files/";
    private String pythonFilePath = "/home/anantha/ARC/AIML_NLP/codePython/nlp1.py";

    @PostMapping("/upload")
    public String handleFileUpload(@RequestParam("file") MultipartFile file) {
        String returnStringinJson = null;
        if (file.isEmpty()) {
            return "File is empty";
        }
        try {
            // Get the bytes of the file
            byte[] bytes = file.getBytes();

            // Create the directory if it doesn't exist
            File directory = new File(uploadPath);
            if (!directory.exists()) {
                directory.mkdirs(); // Create directory including parent directories
            }

            System.out.println("Audio file name: "+file.getOriginalFilename());

            // Create the path where the file will be saved
            Path path = Paths.get(uploadPath + File.separator + file.getOriginalFilename());

            // Save the file
            Files.write(path, bytes);

            try {
                System.out.println("calling python script");
                // Command to execute the Python script
                String[] command = {"python", pythonFilePath, "audio_file_path="+uploadPath+file.getOriginalFilename()};

                // Create ProcessBuilder instance
                ProcessBuilder processBuilder = new ProcessBuilder(command);

                // Start the process
                Process process = processBuilder.start();

                // Read output from Python script
                BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                String line;
                StringBuilder jsonStringBuilder = new StringBuilder();
                int i = 0;
                while ((line = reader.readLine()) != null) {
                    i++;
                    System.out.println(i+" :"+line);
                    if(i == 9){
                        returnStringinJson = line;
                    }
                }

                // Wait for the process to finish
                int exitCode = process.waitFor();
                System.out.println("Python script exited with code " + exitCode);

            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            }

            return returnStringinJson;
        } catch (IOException e) {
            e.printStackTrace();
            return "Failed to process file.";
        }
    }
}
