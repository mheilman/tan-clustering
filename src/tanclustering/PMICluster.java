package tanclustering;
import java.util.*;
import java.io.*;
import java.util.Map.Entry;

// uses commons-io-2.4.jar
import org.apache.commons.io.FileUtils;


public class PMICluster {
	public class Tuple<X, Y>{
		public final X v1;
		public final Y v2;

		public Tuple(X v1, Y v2){
			this.v1 = v1;
			this.v2 = v2;
		} 
	}

	private Map<String, Integer> wordIDs = new HashMap<String, Integer>();

	// mapping from cluster IDs to cluster IDs, to keep track of the hierarchy
	private Map<Integer, Integer> clusterParents = new HashMap<Integer, Integer>();

	// the 0/1 bit to add when walking up the hierarchy from a word to the top-level cluster
	private Map<Integer, String> clusterBits = new HashMap<Integer, String>();

	private List<String> words = new ArrayList<String>();
	private Map<Integer, Integer> wordCounts = new HashMap<Integer, Integer>();
	private Random rng = new Random(1234567890);

	// clusterID -> {docID -> counts}
	private Map<Integer, Map<Integer, Integer> > index = new HashMap<Integer, Map<Integer, Integer> >();

	private int clusterCounter = 0;

	private int batchSize = 0;
	private boolean lower = false;

	private List<Integer> currentBatch = new ArrayList<Integer>();
	private Map<Integer, Map<Integer, Double> > currentBatchScores = new HashMap<Integer, Map<Integer, Double> >();


	public PMICluster(String inputPath, boolean lower, int maxVocabSize, int minWordCount, int batchSize){
		this.batchSize = batchSize;
		this.lower = lower;
		try{
			init(inputPath, maxVocabSize, minWordCount);
		}catch(IOException e){
			e.printStackTrace();
		}
	}


	private void init(String inputPath, int maxVocabSize, int minWordCount) throws IOException{
		makeWordCounts(inputPath, maxVocabSize, minWordCount);

		// make a copy of the list of words, as a queue for making new clusters
		LinkedList<Integer> wordQueue = new LinkedList<Integer>();
		for(String w: words){
			int wordID = wordIDs.get(w);
			wordQueue.add(clusterCounter);
			clusterCounter = wordID;
		}
		clusterCounter++;

		// create sets of documents that each word appears in
		createIndex(inputPath);

		// score potential clusters, starting with the most frequent words.
		// also, remove the batch from the queue
		for(int i = 0; i < batchSize && wordQueue.size() > 0; i++){
			currentBatch.add(wordQueue.pop());
		}
		makePairScores(allPairs(currentBatch));

		while(currentBatch.size() > 1){
			// find the best pair of words/clusters to merge
			Tuple<Integer, Integer> best = findBest();
			int c1 = best.v1;
			int c2 = best.v2;

			// merge the clusters in the index
			merge(c1, c2);

			// remove the merged clusters from the batch, add the new one
			// and the next most frequent word (if available)
			updateBatch(c1, c2, wordQueue);

			System.err.println(idToWord(c1) + " AND " + idToWord(c2) + " WERE MERGED INTO " + clusterCounter + ". " + (currentBatch.size() + wordQueue.size() - 1) + " REMAIN.");

			clusterCounter++;
		}
	}

	private String idToWord(int id){
		if(id < words.size()){
			return words.get(id);
		}else{
			return String.valueOf(id);
		}
	}


	private void createIndex(String inputPath) throws IOException{
		Iterator<String> it = FileUtils.lineIterator(new File(inputPath), "UTF-8");
		String doc;
		Integer docID = Integer.valueOf(0);
		while(it.hasNext()){
			doc = it.next();
			String[] toks = doc.trim().split(" ");
			for(String tok: toks){
				if(lower){
					tok = tok.toLowerCase();
				}
				
				if(!wordIDs.containsKey(tok)){
					continue;
				}
				int wordID = wordIDs.get(tok);

				if(!index.containsKey(wordID)){
					index.put(wordID, new HashMap<Integer, Integer>());
				}

				if(!index.get(wordID).containsKey(docID)){
					index.get(wordID).put(docID, 0);
				}
				int tmpVal = index.get(wordID).get(docID);
				index.get(wordID).put(docID, tmpVal + 1);
			}
			docID = Integer.valueOf(docID + 1);
		}
		System.err.println(docID + " documents were indexed.");
	}


	private <X, Y> List<Tuple<X, X> > allPairs(List<X> collection){
		List<Tuple<X, X> > res = new ArrayList<Tuple<X, X> >();
		for(int i = 0; i < collection.size(); i++){
			for(int j = i + 1; j < collection.size(); j++){
				res.add(new Tuple<X, X>(collection.get(i), collection.get(j)));
			}
		}
		return res;
	}

	private <X, Y> List<Tuple<X, X> > productPairs(List<X> collection1, List<X> collection2){
		List<Tuple<X, X> > res = new ArrayList<Tuple<X, X> >();
		for(int i = 0; i < collection1.size(); i++){
			for(int j = 0; j < collection2.size(); j++){
				res.add(new Tuple<X, X>(collection1.get(i), collection2.get(j)));
			}
		}
		return res;
	}


	private void makePairScores(List<Tuple<Integer, Integer> > pairList){
		for(Tuple<Integer, Integer> tuple: pairList){
			int c1 = tuple.v1;
			int c2 = tuple.v2;

			int paircount = 0;

			Map<Integer, Integer> index1 = index.get(c1);
			Map<Integer, Integer> index2 = index.get(c2);
			if(index1.size() > index2.size()){
				Map<Integer, Integer> tmp = index1;
				index1 = index2;
				index2 = tmp;
			}

			for(Integer docID: index1.keySet()){
				if(index2.containsKey(docID)){
					paircount += index1.get(docID) * index2.get(docID);
				}
			}

			if(!currentBatchScores.containsKey(c1)){
				currentBatchScores.put(c1, new HashMap<Integer, Double>());
			}

			if(paircount == 0){
				currentBatchScores.get(c1).put(c2, Double.NEGATIVE_INFINITY);  // log(0)
			}else{
				double score = Math.log(paircount) - Math.log(wordCounts.get(c1)) - Math.log(wordCounts.get(c2));
				currentBatchScores.get(c1).put(c2, score);
			}
		}
	}

	private Tuple<Integer, Integer> findBest(){
		double bestScore = Double.NEGATIVE_INFINITY;
		List<Tuple<Integer, Integer> > argmaxList = new ArrayList<Tuple<Integer, Integer> >();

		int c1;
		int c2;
		double score;

		for(Entry<Integer, Map<Integer, Double> > entry1: currentBatchScores.entrySet()){
			c1 = entry1.getKey();
			for(Entry<Integer, Double> entry2: entry1.getValue().entrySet()){
				c2 = entry2.getKey();
				score = entry2.getValue();
				if(score > bestScore){
					argmaxList = new ArrayList<>();
					argmaxList.add(new Tuple<Integer, Integer>(c1, c2));
					bestScore = score;
				}else if(score == bestScore){
					argmaxList.add(new Tuple<Integer, Integer>(c1, c2));
				}
			}
		}

		// break ties randomly
		Tuple<Integer, Integer> res = argmaxList.get(rng.nextInt(argmaxList.size()));
		return res;
	}


	private void merge(int c1, int c2){
		clusterParents.put(c1, clusterCounter);
		clusterParents.put(c2, clusterCounter);

		int r = rng.nextInt(2);
		clusterBits.put(c1, String.valueOf(r));  // assign bits randomly
		clusterBits.put(c2, String.valueOf(1 - r));

		// initialize the document counts of the new cluster with the counts
		// for one of the two child clusters.  then, add the counts from the
		// other child cluster
		Map<Integer, Integer> indexNew = index.get(c1);
		index.put(clusterCounter, indexNew);
		for(int docID: index.get(c2).keySet()){
			if(!indexNew.containsKey(docID)){
				indexNew.put(docID, 0);
			}
			int tmpVal = indexNew.get(docID);
			indexNew.put(docID, tmpVal + index.get(c2).get(docID));
		}

		// sum the frequencies of the child clusters
		wordCounts.put(clusterCounter, wordCounts.get(c1) + wordCounts.get(c2));

		// remove merged clusters from the index to save memory
		// (but keep frequencies for words for the final output)
		index.remove(c1);
		index.remove(c2);
		if(c1 > words.size()){
			wordCounts.remove(c1);
		}
		if(c2 > words.size()){
			wordCounts.remove(c2);
		}
	}


	private void updateBatch(int c1, int c2, List<Integer> freqWords){
		// remove the clusters that were merged (and the scored pairs for them)
		List<Integer> oldBatch = currentBatch;
		currentBatch = new ArrayList<Integer>();
		for(Integer c: oldBatch){
			if(c != c1 && c != c2){
				currentBatch.add(c);
			}
		}

		currentBatchScores.remove(c1);
		currentBatchScores.remove(c2);
		for(Map<Integer, Double> d: currentBatchScores.values()){
			d.remove(c1);
			d.remove(c2);
		}

		// find what to add to the current batch
		List<Integer> newItems = new ArrayList<Integer>();
		newItems.add(clusterCounter);
		if(freqWords.size() > 0){
			newItems.add(freqWords.get(0));
			freqWords.remove(0);
		}

		// add to the batch and score the new cluster pairs that result
		makePairScores(productPairs(currentBatch, newItems));
		makePairScores(allPairs(newItems));

		// note: make the scores first with itertools.product
		// (before adding new_items to current_batch) to avoid duplicates
		currentBatch.addAll(newItems);
	}      


	private Comparator<Entry<String, Integer> > mapComparatorByIntValue = new Comparator<Entry<String, Integer> >(){
		public int compare(Entry<String, Integer> o1, Entry<String, Integer> o2){
			return o1.getValue().compareTo(o2.getValue());
		}
	};


	private void makeWordCounts(String inputPath, int maxVocabSize, int minWordCount) throws IOException{
		Map<String, Integer> wordCountsTmp = new HashMap<String, Integer>();

		Iterator<String> it = FileUtils.lineIterator(new File(inputPath), "UTF-8");
		String doc;
		while(it.hasNext()){
			doc = it.next();
			String[] toks = doc.trim().split(" ");
			for(String tok: toks){
				if(lower){
					tok = tok.toLowerCase();
				}

				if(!wordCountsTmp.containsKey(tok)){
					wordCountsTmp.put(tok, 0);
				}
				wordCountsTmp.put(tok, wordCountsTmp.get(tok) + 1);
			}
		}

		int tooRare = minWordCount - 1;
		if(minWordCount > 1 && maxVocabSize > 0){
			System.err.println("maxVocabSize and minWordCount both set.  Ignoring minWordCount.");
		}

		List<Entry<String, Integer> > wordEntries = new ArrayList<Entry<String, Integer> >();
		wordEntries.addAll(wordCountsTmp.entrySet());
		Collections.sort(wordEntries, mapComparatorByIntValue);
		Collections.reverse(wordEntries);
		for(Entry<String, Integer> entry: wordEntries){
			words.add(entry.getKey());
		}

		if(maxVocabSize > 0){
			if(words.size() <= maxVocabSize){
				tooRare = 0;
			}else{
				tooRare = wordCountsTmp.get(words.get(maxVocabSize));
				if(tooRare == wordCountsTmp.get(words.get(0))){
					tooRare += 1;
					System.err.println("max_vocab_size too low.  Using all words that appeared > " + tooRare + " times.");
				}
			}
		}

		// only keep words that occur more frequently than too_rare
		List<String> tmpWords = words;
		words = new ArrayList<String>();
		for(String w: tmpWords){
			if(wordCountsTmp.get(w) > tooRare){
				// can stop after the first too rare word is reached because the list is sorted
				words.add(w);
			}
		}

		System.err.println("Created vocabulary with the " + words.size() + " words that occurred at least " + (tooRare + 1) + " times.");

		Set<String> wordSet = new HashSet<String>();
		int wordID = 0;
		for(String w: words){
			wordIDs.put(w, wordID);
			wordCounts.put(wordID, wordCountsTmp.get(w));
			wordID++;
		}
	}


	private String getBitstring(String w){
		// walk up the cluster hierarchy until there is no parent cluster
		int curCluster = wordIDs.get(w);
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
				pw.println(w + "\t" + getBitstring(w) + "\t" + wordCounts.get(wordIDs.get(w)));
			}
		}catch(IOException e){
			e.printStackTrace();
		}
	}


	public static void main(String[] args) {
		if(args.length != 6){
			System.err.println("args: <inputPath> <outputPath> <lowercase (true/false)> <minWordCount> <maxVocabSize> <batchSize>");
			System.exit(0);
		}

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
						   "maxVocabSize=" + maxVocabSize + "\n" + 
						   "batchSize=" + batchSize);

		PMICluster c = new PMICluster(inputPath, lower, maxVocabSize, minWordCount, batchSize);
		
		c.saveClusters(outputPath);
	}
}
