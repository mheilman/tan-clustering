package com.github.mheilman.tanclustering;
import java.util.*;
import java.io.*;
import java.util.Map.Entry;
import java.util.concurrent.*;
import java.util.logging.Logger;

// uses commons-io-2.4.jar
import org.apache.commons.io.FileUtils;

//TODO use trove, COLT, or apache commons collections for hash maps (?)

public class PMICluster {
	private final static Logger LOGGER = Logger.getLogger(PMICluster.class.getName()); 

	public class Tuple<X, Y>{
		public final X v1;
		public final Y v2;

		public Tuple(X v1, Y v2){
			this.v1 = v1;
			this.v2 = v2;
		}
		public String toString(){
			return "(" + v1 + ", " + v2 + ")";
		}
	}

	private Map<String, Integer> wordIDs = new HashMap<String, Integer>();

	// mapping from cluster IDs to cluster IDs, to keep track of the hierarchy
	private Map<Integer, Integer> clusterParents = new HashMap<Integer, Integer>();

	// the 0/1 bit to add when walking up the hierarchy from a word to the top-level cluster
	private Map<Integer, Integer> clusterBits = new HashMap<Integer, Integer>();

	private List<String> words = new ArrayList<String>();
	private Map<Integer, Long> wordCounts = new HashMap<Integer, Long>();
	private Random rng = new Random(1234567890);

	// clusterID -> {docID -> counts}
	private Map<Integer, Map<Integer, Long> > index = new HashMap<Integer, Map<Integer, Long> >();

	private Integer clusterCounter = 0;

	private int batchSize = 0;
	private boolean lower = false;

	private List<Integer> currentBatch = new ArrayList<Integer>();
	private Map<Integer, Map<Integer, Double> > currentBatchScores = new HashMap<Integer, Map<Integer, Double> >();
	
	private int numThreads;
	private ExecutorService executor = null;

	public class PairScorerCallable implements Callable {
		public List<Tuple<Integer, Integer> > pairList = null;

		public PairScorerCallable(List<Tuple<Integer, Integer> > pairList){
			this.pairList = pairList;
		}

		public List<Double> call() {
			List<Double> res = new ArrayList<Double>();
			for(Tuple<Integer, Integer> tuple: pairList){
				Integer c1 = tuple.v1;
				Integer c2 = tuple.v2;

				long paircount = 0;

				Map<Integer, Long> index1 = index.get(c1);
				Map<Integer, Long> index2 = index.get(c2);
				if(index1.size() > index2.size()){
					Map<Integer, Long> tmp = index1;
					index1 = index2;
					index2 = tmp;
				}

				for(Integer docID: index1.keySet()){
					if(index2.containsKey(docID)){
						paircount += index1.get(docID) * index2.get(docID);
					}
				}

				if(paircount == 0){
					res.add(Double.NEGATIVE_INFINITY);  // log(0)
				}else{
					double score = Math.log(paircount) - Math.log(wordCounts.get(c1)) - Math.log(wordCounts.get(c2));
					assert !Double.isNaN(score);
					res.add(score);
				}
			}

			assert res.size() == pairList.size();

			return res;
		}
	}


	public PMICluster(String inputPath, boolean lower, int maxVocabSize, int minWordCount, int batchSize, int numThreads){
		this.batchSize = batchSize;
		this.lower = lower;
		this.numThreads = numThreads;
		try{
			init(inputPath, maxVocabSize, minWordCount);
		}catch(IOException e){
			e.printStackTrace();
		}
	}


	private void init(String inputPath, int maxVocabSize, int minWordCount) throws IOException{
		executor = Executors.newFixedThreadPool(numThreads);

		makeWordCounts(inputPath, maxVocabSize, minWordCount);

		// make a copy of the list of words, as a queue for making new clusters
		LinkedList<Integer> wordStack = new LinkedList<Integer>();
		for(String w: words){
			Integer wordID = wordIDs.get(w);
			wordStack.add(wordID);
			clusterCounter = wordID;
		}
		clusterCounter++;

		// create sets of documents that each word appears in
		LOGGER.info("creating index...");
		createIndex(inputPath);

		// score potential clusters, starting with the most frequent words.
		// also, remove the batch from the queue
		LOGGER.info("computing initial scores...");
		for(int i = 0; i <= batchSize && wordStack.size() > 0; i++){
			currentBatch.add(wordStack.pop());
		}
		makePairScores(allPairs(currentBatch));

		LOGGER.info("clustering...");
		while(currentBatch.size() > 1){
			// find the best pair of words/clusters to merge
			Tuple<Integer, Integer> best = findBest();
			Integer c1 = best.v1;
			Integer c2 = best.v2;

			// merge the clusters in the index
			merge(c1, c2);

			// remove the merged clusters from the batch, add the new one
			// and the next most frequent word (if available)
			updateBatch(c1, c2, wordStack);

			LOGGER.info(idToWord(c1) + " AND " + idToWord(c2) + " WERE MERGED INTO " + clusterCounter + ". CURRENTBATCH: " + currentBatch.size() + " WORDSTACK:" + wordStack.size());

			clusterCounter++;
		}

		executor.shutdownNow();
	}

	private String idToWord(Integer id){
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
			if(docID % 10000 == 0){
				LOGGER.info("Indexing document " + docID + ".");
			}

			doc = it.next();
			String[] toks = doc.trim().split(" ");
			for(String tok: toks){
				if(lower){
					tok = tok.toLowerCase();
				}
				
				if(!wordIDs.containsKey(tok)){
					continue;
				}
				Integer wordID = wordIDs.get(tok);

				if(!index.containsKey(wordID)){
					index.put(wordID, new HashMap<Integer, Long>());
				}

				if(!index.get(wordID).containsKey(docID)){
					index.get(wordID).put(docID, 0L);
				}
				Long tmpVal = index.get(wordID).get(docID);
				index.get(wordID).put(docID, tmpVal + 1L);
			}
			docID = Integer.valueOf(docID + 1);
		}
		LOGGER.info(docID + " documents were indexed.");
	}


	private List<Tuple<Integer, Integer> > allPairs(List<Integer> collection){
		List<Tuple<Integer, Integer> > res = new ArrayList<Tuple<Integer, Integer> >();
		for(int i = 0; i < collection.size(); i++){
			for(int j = i + 1; j < collection.size(); j++){
				Integer v1 = collection.get(i);
				Integer v2 = collection.get(j);
				if(v1 > v2){
					Integer tmp = v1;
					v1 = v2;
					v2 = tmp;
				}
				res.add(new Tuple<Integer, Integer>(v1, v2));
			}
		}
		return res;
	}

	private List<Tuple<Integer, Integer> > newPairs(List<Integer> newItems){
		List<Tuple<Integer, Integer> > res = new ArrayList<Tuple<Integer, Integer> >();
		for(int i = 0; i < currentBatch.size(); i++){
			for(int j = 0; j < newItems.size(); j++){
				Integer v1 = currentBatch.get(i);
				Integer v2 = newItems.get(j);
				if(v1 > v2){
					Integer tmp = v1;
					v1 = v2;
					v2 = tmp;
				}
				res.add(new Tuple<Integer, Integer>(v1, v2));
			}
		}

		if(newItems.size() > 1){
			res.addAll(allPairs(newItems));
		}

		// LOGGER.info("newPairs res: " + res.size() + " currentBatch.size():" + currentBatch.size());
		return res;
	}


	private void makePairScores(List<Tuple<Integer, Integer> > pairList){
		List<Future<List<Double> > > task_results = new ArrayList<Future<List<Double> > >();

		int n = pairList.size();

		// LOGGER.info("makePairScores pairList.size(): " + pairList.size());
		if(n == 0){
			return;
		}

		PairScorerCallable task;
		double chunkSize = (double) n / numThreads;
		int startIdx = 0;
		int endIdx;
		for(int i = 0; i < numThreads; i++){
			if(i == numThreads - 1){
				endIdx = n;
			}else{
				endIdx = (int) (chunkSize * (i + 1));
			}
			task = new PairScorerCallable(pairList.subList(startIdx, endIdx));
			task_results.add(executor.submit(task));
			startIdx = endIdx;
		}

		List<Double> scores = new ArrayList<Double>();
		try{
			for(int i = 0; i < numThreads; i++){
				scores.addAll(task_results.get(i).get());
			}
		}catch(ExecutionException e){
			LOGGER.severe("Exception caught in makePairScores:\n" + e);
			System.exit(1);
		}catch(InterruptedException e){
			LOGGER.severe("Exception caught in makePairScores:\n" + e);
			System.exit(1);
		}

		assert scores.size() == pairList.size();

		for(int i = 0; i < pairList.size(); i++){
			Tuple<Integer, Integer> pair = pairList.get(i);
			Integer c1 = pair.v1;
			Integer c2 = pair.v2;

			if(!currentBatchScores.containsKey(c1)){
				currentBatchScores.put(c1, new HashMap<Integer, Double>());
			}

			currentBatchScores.get(c1).put(c2, scores.get(i));
		}

	}


	private Tuple<Integer, Integer> findBest(){

		double bestScore = Double.NEGATIVE_INFINITY;

		List<Integer> argmaxListC1 = new ArrayList<Integer>();
		List<Integer> argmaxListC2 = new ArrayList<Integer>();

		Integer c1;
		Integer c2;
		double score;

		// int n = 0;
		// int m = 0;
		for(Entry<Integer, Map<Integer, Double> > entry1: currentBatchScores.entrySet()){
			c1 = entry1.getKey();
			for(Entry<Integer, Double> entry2: entry1.getValue().entrySet()){
				c2 = entry2.getKey();
				score = entry2.getValue();
				if(score >= bestScore){
					if(score > bestScore){
						argmaxListC1.clear();
						argmaxListC2.clear();
						bestScore = score;
					}
					argmaxListC1.add(c1);
					argmaxListC2.add(c2);
					// m++;
				}
				// n++;
			}
		}

		// LOGGER.info("findBest: m = " + m + " n = " + n);
		// if(n < 10){
		// 	LOGGER.info(currentBatchScores.toString());
		// }

		// break ties randomly
		int r = rng.nextInt(argmaxListC1.size());
		Tuple<Integer, Integer> res = new Tuple<Integer, Integer>(argmaxListC1.get(r), argmaxListC2.get(r));
		return res;
	}


	private void merge(Integer c1, Integer c2){
		clusterParents.put(c1, clusterCounter);
		clusterParents.put(c2, clusterCounter);

		int r = rng.nextInt(2);
		clusterBits.put(c1, r);  // assign bits randomly
		clusterBits.put(c2, 1 - r);

		// initialize the document counts of the new cluster with the counts
		// for one of the two child clusters.  then, add the counts from the
		// other child cluster
		Map<Integer, Long> indexNew = index.get(c1);
		index.put(clusterCounter, indexNew);
		for(Integer docID: index.get(c2).keySet()){
			if(!indexNew.containsKey(docID)){
				indexNew.put(docID, 0L);
			}
			Long tmpVal = indexNew.get(docID);
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


	private void updateBatch(Integer c1, Integer c2, LinkedList<Integer> wordStack){
		// remove the clusters that were merged (and the scored pairs for them)
		List<Integer> oldBatch = currentBatch;
		currentBatch = new ArrayList<Integer>();
		for(Integer c: oldBatch){
			if(c != c1 && c != c2){
				currentBatch.add(c);
			}
		}

		assert currentBatch.size() == oldBatch.size() - 2;

		currentBatchScores.remove(c1);
		currentBatchScores.remove(c2);
		for(Map<Integer, Double> d: currentBatchScores.values()){
			d.remove(c1);
			d.remove(c2);
		}

		// find what to add to the current batch
		List<Integer> newItems = new ArrayList<Integer>();
		newItems.add(clusterCounter);
		if(wordStack.size() > 0){
			newItems.add(wordStack.pop());
		}

		assert newItems.size() > 0;
		// LOGGER.info("NEW ITEMS SIZE:" + newItems.size());
		
		// add to the batch and score the new cluster pairs that result
		makePairScores(newPairs(newItems));

		// note: make the scores first with itertools.product
		// (before adding new_items to current_batch) to avoid duplicates
		currentBatch.addAll(newItems);
	}      


	private Comparator<Entry<String, Long> > mapComparatorByIntValue = new Comparator<Entry<String, Long> >(){
		public int compare(Entry<String, Long> o1, Entry<String, Long> o2){
			return o1.getValue().compareTo(o2.getValue());
		}
	};


	private void makeWordCounts(String inputPath, int maxVocabSize, int minWordCount) throws IOException{
		Map<String, Long> wordCountsTmp = new HashMap<String, Long>();

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
					wordCountsTmp.put(tok, 0L);
				}
				wordCountsTmp.put(tok, wordCountsTmp.get(tok) + 1);
			}
		}

		long tooRare = minWordCount - 1;
		if(minWordCount > 1 && maxVocabSize > 0){
			LOGGER.info("maxVocabSize and minWordCount both set.  Ignoring minWordCount.");
		}

		List<Entry<String, Long> > wordEntries = new ArrayList<Entry<String, Long> >();
		wordEntries.addAll(wordCountsTmp.entrySet());
		Collections.sort(wordEntries, mapComparatorByIntValue);
		Collections.reverse(wordEntries);
		for(Entry<String, Long> entry: wordEntries){
			words.add(entry.getKey());
		}

		if(maxVocabSize > 0){
			if(words.size() <= maxVocabSize){
				tooRare = 0;
			}else{
				tooRare = wordCountsTmp.get(words.get(maxVocabSize));
				if(tooRare == wordCountsTmp.get(words.get(0))){
					tooRare += 1;
					LOGGER.info("max_vocab_size too low.  Using all words that appeared > " + tooRare + " times.");
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

		LOGGER.info("Created vocabulary with the " + words.size() + " words that occurred at least " + (tooRare + 1) + " times.");

		Integer wordID = 0;
		for(String w: words){
			wordIDs.put(w, wordID);
			wordCounts.put(wordID, wordCountsTmp.get(w));
			wordID++;
		}
	}


	private String getBitstring(String w){
		// walk up the cluster hierarchy until there is no parent cluster
		Integer curCluster = wordIDs.get(w);
		String bitstring = "";
		while(clusterParents.containsKey(curCluster)){
			bitstring = String.valueOf(clusterBits.get(curCluster)) + bitstring;
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
			pw.close();
		}catch(IOException e){
			e.printStackTrace();
		}
	}


	public static void main(String[] args) {
		if(args.length != 7){
			System.err.println("args: <inputPath> <outputPath> <lowercase (true/false)> <minWordCount> <maxVocabSize> <batchSize> <numThreads>");
			System.exit(0);
		}

		String inputPath = args[0];
		String outputPath = args[1];
		boolean lower = Boolean.parseBoolean(args[2]);
		int minWordCount = Integer.parseInt(args[3]);
		int maxVocabSize = Integer.parseInt(args[4]);
		int batchSize = Integer.parseInt(args[5]);
		int numThreads = Integer.parseInt(args[6]);

		LOGGER.info("inputPath=" + inputPath + "\n" +
					"outputPath=" + outputPath + "\n" +
					"lower=" + lower + "\n" +
					"minWordCount=" + minWordCount + "\n" +
					"maxVocabSize=" + maxVocabSize + "\n" + 
					"batchSize=" + batchSize + "\n" +
					"numThreads=" + numThreads);

		PMICluster c = new PMICluster(inputPath, lower, maxVocabSize, minWordCount, batchSize, numThreads);
		
		c.saveClusters(outputPath);
	}
}
