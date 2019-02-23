import edu.stanford.nlp.coref.CorefCoreAnnotations;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.io.*;
import java.io.IOException;


/**
 * Created by Mayanka on 13-Jun-16.
 */
public class Main {
    static String slamp="";
    static String sbed="";
    static String schair="";
    static int lamp=0;static int bed=0;static int chair=0;
    public static void main(String args[]) {

        String path="C:\\Users\\anves\\Downloads\\BigDataApplications\\ProjectDataset\\Flickr8k\\Flickr8k.token.txt";
        try {
            File file = new File(path);
            PrintWriter writer = new PrintWriter("the-file-name.txt", "UTF-8");
            BufferedReader br = new BufferedReader(new FileReader(file));

            String st;
            String lemma;
           // st=br.readLine();
           while ((st = br.readLine()) != null) {
                lemma = NLP(st);
                writer.println(lemma);
            }
            writer.close();
        }
        catch (Exception e){
            System.out.println(e.toString());
        }
        System.out.println("Image Statistics");
        System.out.println("Chair : "+chair+" images =>" +schair);
        System.out.println("Bed : "+bed+" images =>" +sbed);
        System.out.println("Lamp : "+lamp+" images =>" +slamp);

    }
    public static String NLP(String data) throws Exception{
        String Image=data.substring(0,data.indexOf(".jpg")+4);
        data=data.substring(data.indexOf(".jpg")+7);
        String Sentencelemma="";
        // creates a StanfordCoreNLP object, with POS tagging, lemmatization, NER, parsing, and coreference resolution
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        // read some text in the text variable
        String text = data; // Add your text here!

// create an empty Annotation just with the given text
        Annotation document = new Annotation(text);

// run all Annotators on this text
        pipeline.annotate(document);

        // these are all the sentences in this document
// a CoreMap is essentially a Map that uses class objects as keys and has values with custom types
        List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);

        for (CoreMap sentence : sentences) {
            // traversing the words in the current sentence
            // a CoreLabel is a CoreMap with additional token-specific methods
            for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {

                /*System.out.println("\n" + token);

                // this is the text of the token
                String word = token.get(CoreAnnotations.TextAnnotation.class);
                System.out.println("Text Annotation");
                System.out.println(token + ":" + word);
                // this is the POS tag of the token
*/
                String lemma = token.get(CoreAnnotations.LemmaAnnotation.class);
                //System.out.println("Lemma Annotation");
                //System.out.println(lemma);
                Sentencelemma=Sentencelemma+lemma+" ";
                // this is the Lemmatized tag of the token
                try{

                    if(lemma.equals("bed")){
                        bed++;
                        sbed=sbed+Image+",";
                        //System.out.println("check");
                        String spath="C:\\Users\\anves\\Downloads\\BigDataApplications\\ProjectDataset\\Flickr8k\\Flicker8k_Dataset";
                        String dpath="C:\\Users\\anves\\Downloads\\BigDataApplications\\ProjectDataset\\Bed";
                        Path sourceDirectory = Paths.get(spath+"\\"+Image);
                        Path targetDirectory = Paths.get(dpath+"\\"+Image);

                        //copy source to target using Files Class
                        Files.copy(sourceDirectory, targetDirectory);
                    }
                    if(lemma.equals("chair")){
                        chair++;
                        schair=schair+Image+",";
                        //System.out.println("check");
                        String spath="C:\\Users\\anves\\Downloads\\BigDataApplications\\ProjectDataset\\Flickr8k\\Flicker8k_Dataset";
                        String dpath="C:\\Users\\anves\\Downloads\\BigDataApplications\\ProjectDataset\\Chair";
                        Path sourceDirectory = Paths.get(spath+"\\"+Image);
                        Path targetDirectory = Paths.get(dpath+"\\"+Image);

                        //copy source to target using Files Class
                        Files.copy(sourceDirectory, targetDirectory);
                    }
                    if(lemma.equals("lamp")){
                        lamp++;
                        slamp=slamp+Image+"";
                        //System.out.println("check");
                        String spath="C:\\Users\\anves\\Downloads\\BigDataApplications\\ProjectDataset\\Flickr8k\\Flicker8k_Dataset";
                        String dpath="C:\\Users\\anves\\Downloads\\BigDataApplications\\ProjectDataset\\Lamp";
                        Path sourceDirectory = Paths.get(spath+"\\"+Image);
                        Path targetDirectory = Paths.get(dpath+"\\"+Image);

                        //copy source to target using Files Class
                        Files.copy(sourceDirectory, targetDirectory);
                    }
                }
                catch (Exception e){}

            }
        }
        return Sentencelemma;
    }
}
