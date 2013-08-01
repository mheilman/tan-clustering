package tanclustering;
import java.util.*;
import java.io.*;

// uses commons-io-2.4.jar
import org.apache.commons.io.FileUtils;


public class PMICluster {
	private Map<String, Integer> wordIndex = new HashMap<String, Integer>();
	private Map<Integer, Integer> clusterParents = new HashMap<Integer, Integer>();
	private Map<Integer, String> clusterBits = new HashMap<Integer, String>();
	private List<String> words = new ArrayList<String>();
	private Map<String, Integer> wordCounts = new HashMap<String, Integer>();
	private Random rng = new Random(1234567890);

	public PMICluster(String inputPath, boolean lower, int maxVocabSize, int minWordCount, int batchSize){

	}


	private Map<String, Integer> makeWordCounts(String inputPath, boolean lower, int maxVocabSize, int minWordCount) throws IOException{
		Iterator<String> it = FileUtils.lineIterator(new File(inputPath), "UTF-8");
		//TODO
		return null;
	}


	private String getBitstring(String w){
		// walk up the cluster hierarchy until there is no parent cluster
        int curCluster = wordIndex.get(w);
        String bitstring = "";
        while(clusterParents.containsKey(curCluster)){
            bitstring = clusterBits.get(curCluster) + bitstring;
            curCluster = clusterParents.get(curCluster);
        }
        return bitstring;
	}

	public void saveClusters(String outputPath){
		try{
			PrintWriter pw = new PrintWriter(new File(outputPath));
			for(String w: words){
                pw.println(w + "\t" + getBitstring(w) + "\t" + wordCounts.get(w));
            }
		}catch(IOException e){
			e.printStackTrace();
		}
	}

	public static void main(String[] args) {
		
		String inputPath = args[0];
		String outputPath = args[1];
		boolean lower = Boolean.parseBoolean(args[2]);
		int minWordCount = Integer.parseInt(args[3]);
		int maxVocabSize = Integer.parseInt(args[4]);
		int batchSize = Integer.parseInt(args[5]);

		System.err.println("inputPath=" + inputPath + "\n" +
						   "outputPath=" + outputPath + "\n" +
						   "lower=" + lower + "\n" +
						   "minWordCount=" + minWordCount + "\n" +
						   "maxVocabSize=" + maxVocabSize);

		PMICluster c = new PMICluster(inputPath, lower, maxVocabSize, minWordCount, batchSize);
		
		c.saveClusters(outputPath);
	}
}
