package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"
)

type Sequence []int8
type PAFPert []float64

type ChunkResult struct {
	BranchIndex int
	Seq         Sequence
	PafPart     PAFPert
}

type SolutionPair struct {
	A Sequence
	B Sequence
}

const (
	mBits                 = 16
	bucketLimit           = 100
	numSequencesPerWorker = 50000
	floatTolerance        = 1e-9
	maxRetriesPerOrder    = 5
)

func rollSlice(seq Sequence, k int) Sequence {
	n := len(seq)
	if n == 0 {
		return seq
	}
	k = k % n
	if k < 0 {
		k += n
	}
	rolled := make(Sequence, n)
	copy(rolled, seq[k:])
	copy(rolled[n-k:], seq[:k])
	return rolled
}

func sumInt8(slice Sequence) int {
	sum := 0
	for _, v := range slice {
		sum += int(v)
	}
	return sum
}

func calculatePAFDirect(seq Sequence, length int) []float64 {
	n := len(seq)
	if length > n || length <= 0 {
		length = n
	}
	paf := make([]float64, length)
	if n == 0 {
		return paf
	}
	paf[0] = float64(n)
	for k := 1; k < length; k++ {
		sum := 0
		rolledSeq := rollSlice(seq, -k)
		for i := 0; i < n; i++ {
			sum += int(seq[i] * rolledSeq[i])
		}
		paf[k] = float64(sum)
	}
	return paf
}

func getBranchIndex(pafPart PAFPert, mBits int, offset float64) int {
	N := 0
	limit := mBits
	if limit > len(pafPart) {
		limit = len(pafPart)
	}
	for i := 0; i < limit; i++ {
		adjustedPAF := pafPart[i] + offset
		if offset == 0 {
			if pafPart[i] > floatTolerance {
				N += (1 << i)
			}
		} else {
			if adjustedPAF < -floatTolerance {
				N += (1 << i)
			}
		}
	}
	return N
}

func verifySolution(a, b Sequence, nOrder int, out io.Writer) bool {
	vCheck := nOrder / 2
	fmt.Fprintf(out, "\nVerification for order %d (v=%d):\n", nOrder, vCheck)
	if len(a) != vCheck || len(b) != vCheck {
		fmt.Fprintf(out, "  ERROR: Sequence length incorrect. len(a)=%d, len(b)=%d, expected %d\n", len(a), len(b), vCheck)
		return false
	}
	sumA := sumInt8(a)
	sumB := sumInt8(b)
	if math.Abs(math.Abs(float64(sumA))-1.0) > floatTolerance || math.Abs(math.Abs(float64(sumB))-1.0) > floatTolerance {
		fmt.Fprintf(out, "  WARNING: Sum constraint potentially failed: sum(a)=%d, sum(b)=%d (target |sum|=1)\n", sumA, sumB)
	}
	kTarget := 0
	if vCheck%2 != 0 {
		kTarget = (vCheck - 1) / 2
	} else {
		kTarget = (vCheck - 1) / 2
		fmt.Fprintf(out, "  INFO: Verification uses K=%d based on floor((v-1)/2) for even v=%d.\n", kTarget, vCheck)
	}
	k1Actual := 0
	k2Actual := 0
	for _, val := range a {
		if val == -1 {
			k1Actual++
		}
	}
	for _, val := range b {
		if val == -1 {
			k2Actual++
		}
	}
	if k1Actual != kTarget || k2Actual != kTarget {
		fmt.Fprintf(out, "  WARNING: K constraint failed: k1=%d, k2=%d (target %d)\n", k1Actual, k2Actual, kTarget)
	}
	pafA := calculatePAFDirect(a, vCheck)
	pafB := calculatePAFDirect(b, vCheck)
	correct := true
	targetSum := -2.0
	fmt.Fprintf(out, "  PAF Check (k=1 to %d):\n", vCheck-1)
	maxDiff := 0.0
	worstK := -1
	for k := 1; k < vCheck; k++ {
		currentSum := pafA[k] + pafB[k]
		diff := math.Abs(currentSum - targetSum)
		if diff > maxDiff {
			maxDiff = diff
			worstK = k
		}
		if diff > floatTolerance {
			fmt.Fprintf(out, "    FAIL at k=%d: PAF(a)=%.4f, PAF(b)=%.4f, Sum=%.4f != %.1f (Diff: %.2e)\n", k, pafA[k], pafB[k], currentSum, targetSum, diff)
			correct = false
		}
	}
	if correct {
		fmt.Fprintf(out, "  PAF condition SUCCESSFUL (Max difference %.2e at k=%d)\n", maxDiff, worstK)
	} else {
		fmt.Fprintf(out, "  PAF condition FAILED (Max difference %.2e at k=%d)\n", maxDiff, worstK)
	}
	return correct
}

func processChunk(workerID int, numSequences int, kTarget int, sTarget int, vLen int, zLen int, mBits int, pafOffset float64, resultsChan chan<- ChunkResult, wg *sync.WaitGroup, baseSeed int64) {
	defer wg.Done()
	r := rand.New(rand.NewSource(baseSeed + int64(workerID)))
	seqCount := 0
	attempts := 0
	maxAttempts := numSequences * 1000
	indices := make([]int, vLen)
	for i := range indices {
		indices[i] = i
	}
	for seqCount < numSequences && attempts < maxAttempts {
		attempts++
		seq := make(Sequence, vLen)
		for i := range seq {
			seq[i] = 1
		}
		r.Shuffle(vLen, func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
		for i := 0; i < kTarget; i++ {
			seq[indices[i]] = -1
		}

		currentSum := vLen - 2*kTarget
		if currentSum == sTarget {
			pafFull := calculatePAFDirect(seq, zLen)
			if len(pafFull) < zLen {
				continue
			}
			pafPartForBranching := PAFPert(pafFull[1:zLen])
			branchIdx := getBranchIndex(pafPartForBranching, mBits, pafOffset)
			seqCopy := make(Sequence, vLen)
			copy(seqCopy, seq)
			pafPartCopy := make(PAFPert, len(pafPartForBranching))
			copy(pafPartCopy, pafPartForBranching)
			resultsChan <- ChunkResult{branchIdx, seqCopy, pafPartCopy}
			seqCount++
		} else if attempts == 1 {

			log.Printf("[Worker %d] Warning: Generated k=%d implies sum=%d, but s_target=%d. Check parameter consistency (V=%d).", workerID, kTarget, currentSum, sTarget, vLen)
		}
	}
}

func collectResults(resultsChan <-chan ChunkResult, seqDict map[int][]Sequence, pafDict map[int][]PAFPert, wgCollector *sync.WaitGroup) {
	defer wgCollector.Done()
	for result := range resultsChan {
		if len(seqDict[result.BranchIndex]) < bucketLimit {
			seqDict[result.BranchIndex] = append(seqDict[result.BranchIndex], result.Seq)
			pafDict[result.BranchIndex] = append(pafDict[result.BranchIndex], result.PafPart)
		}
	}
}

func runForOrder(nOrder int, writer *bufio.Writer, attemptNum int) bool {
	startTime := time.Now()
	v := nOrder / 2
	kTarget := 0
	sTarget := 0

	if v%2 != 0 {
		kTarget = (v - 1) / 2
		sTarget = v - 2*kTarget
		if math.Abs(float64(sTarget)) != 1.0 {

			fmt.Printf("[%d, Att %d] INFO: Calculated S_TARGET = %d for odd v=%d. Expected |S_TARGET|=1.\n", nOrder, attemptNum, sTarget, v)
		}
	} else {
		kTarget = (v - 1) / 2
		sTarget = v - 2*kTarget

		fmt.Printf("[%d, Att %d] INFO: N_ORDER=%d leads to even v=%d. Using K=%d based on floor((v-1)/2), S_TARGET=%d.\n", nOrder, attemptNum, nOrder, v, kTarget, sTarget)
	}
	s0 := -2.0
	zLen := (v + 1) / 2
	if attemptNum == 1 {
		fmt.Printf("[%d] Parameters: v=%d, k target=%d, s target=%d, s0=%.1f, Z=%d\n", nOrder, v, kTarget, sTarget, s0, zLen)
		fmt.Printf("[%d] Tree Method: m_bits=%d, bucket_limit=%d\n", nOrder, mBits, bucketLimit)
	}
	numWorkers := runtime.NumCPU()
	if attemptNum == 1 {
		fmt.Printf("[%d] Using %d CPU cores.\n", nOrder, numWorkers)
	}
	baseSeed := time.Now().UnixNano() + int64(attemptNum)
	fmt.Printf("[%d, Att %d] Generating sequences for A...\n", nOrder, attemptNum)
	aDict := make(map[int][]Sequence)
	aaDict := make(map[int][]PAFPert)
	resultsChanA := make(chan ChunkResult, numWorkers*10)
	var wgA sync.WaitGroup
	var wgCollectorA sync.WaitGroup
	wgCollectorA.Add(1)
	go collectResults(resultsChanA, aDict, aaDict, &wgCollectorA)
	wgA.Add(numWorkers)
	for i := 0; i < numWorkers; i++ {
		go processChunk(i, numSequencesPerWorker, kTarget, sTarget, v, zLen, mBits, 0.0, resultsChanA, &wgA, baseSeed)
	}
	wgA.Wait()
	close(resultsChanA)
	wgCollectorA.Wait()
	timeA := time.Now()
	totalGeneratedA := 0
	for _, bucket := range aDict {
		totalGeneratedA += len(bucket)
	}
	fmt.Printf("[%d, Att %d] Generated %d 'a' seqs in %.2fs. Populated %d buckets.\n", nOrder, attemptNum, totalGeneratedA, timeA.Sub(startTime).Seconds(), len(aDict))
	fmt.Printf("[%d, Att %d] Generating sequences for B...\n", nOrder, attemptNum)
	bDict := make(map[int][]Sequence)
	bbDict := make(map[int][]PAFPert)
	resultsChanB := make(chan ChunkResult, numWorkers*10)
	var wgB sync.WaitGroup
	var wgCollectorB sync.WaitGroup
	wgCollectorB.Add(1)
	go collectResults(resultsChanB, bDict, bbDict, &wgCollectorB)
	wgB.Add(numWorkers)
	for i := 0; i < numWorkers; i++ {
		go processChunk(i+numWorkers, numSequencesPerWorker, kTarget, sTarget, v, zLen, mBits, -s0, resultsChanB, &wgB, baseSeed)
	}
	wgB.Wait()
	close(resultsChanB)
	wgCollectorB.Wait()
	timeB := time.Now()
	totalGeneratedB := 0
	for _, bucket := range bDict {
		totalGeneratedB += len(bucket)
	}
	fmt.Printf("[%d, Att %d] Generated %d 'b' seqs in %.2fs. Populated %d buckets.\n", nOrder, attemptNum, totalGeneratedB, timeB.Sub(timeA).Seconds(), len(bDict))
	fmt.Printf("[%d, Att %d] Comparing sequences...\n", nOrder, attemptNum)
	foundPairs := []SolutionPair{}
	comparisons := 0
	checkLen := zLen - 1
	commonKeys := 0
	comparisonStartTime := time.Now()
	for k, bucketA := range aaDict {
		bucketB, exists := bbDict[k]
		if !exists || len(bucketA) == 0 || len(bucketB) == 0 {
			continue
		}
		commonKeys++
		stopBucket := false
		for i, pafA := range bucketA {
			if len(pafA) != checkLen {
				fmt.Printf("[%d, Att %d] WARNING: PAF(a) part length mismatch bucket %d, i=%d. len=%d, expected %d. Skipping.\n", nOrder, attemptNum, k, i, len(pafA), checkLen)
				continue
			}
			for j, pafB := range bucketB {
				comparisons++
				if len(pafB) != checkLen {
					fmt.Printf("[%d, Att %d] WARNING: PAF(b) part length mismatch bucket %d, j=%d. len=%d, expected %d. Skipping.\n", nOrder, attemptNum, k, j, len(pafB), checkLen)
					continue
				}
				match := true
				for idx := 0; idx < checkLen; idx++ {
					if math.Abs(pafA[idx]+pafB[idx]-s0) > floatTolerance {
						match = false
						break
					}
				}
				if match {
					seqA := aDict[k][i]
					seqB := bDict[k][j]
					fmt.Printf("\n[%d, Att %d] Found potential pair in bucket %d! (Indices i=%d, j=%d)\n", nOrder, attemptNum, k, i, j)
					foundPairs = append(foundPairs, SolutionPair{A: seqA, B: seqB})
					stopBucket = true
					break
				}
			}
			if stopBucket {
				break
			}
		}
	}
	timeCompare := time.Now()
	fmt.Printf("[%d, Att %d] Comparison finished in %.2fs. Checked %d common buckets. Total comparisons: %d\n", nOrder, attemptNum, timeCompare.Sub(comparisonStartTime).Seconds(), commonKeys, comparisons)
	pairFound := len(foundPairs) > 0
	if pairFound {
		fmt.Printf("[%d, Att %d] Found %d potential solution pair(s).\n", nOrder, attemptNum, len(foundPairs))
		firstPair := foundPairs[0]
		fmt.Printf("[%d, Att %d] Verifying first potential solution pair...\n", nOrder, attemptNum)
		isVerified := verifySolution(firstPair.A, firstPair.B, nOrder, os.Stdout)
		if isVerified {
			fmt.Printf("[%d, Att %d] Verification result: SUCCESS\n", nOrder, attemptNum)
			aStr := strings.Trim(strings.Join(strings.Fields(fmt.Sprint(firstPair.A)), ","), "[]")
			bStr := strings.Trim(strings.Join(strings.Fields(fmt.Sprint(firstPair.B)), ","), "[]")
			fmt.Fprintf(writer, "Order: %d\n", nOrder)
			fmt.Fprintf(writer, "a = [%s]\n", aStr)
			fmt.Fprintf(writer, "b = [%s]\n", bStr)
			fmt.Fprintln(writer)
			writer.Flush()
		} else {
			fmt.Printf("[%d, Att %d] Verification result: FAILED\n", nOrder, attemptNum)
		}
	} else {
		fmt.Printf("[%d, Att %d] No potential solution pairs found matching the first %d PAF conditions.\n", nOrder, attemptNum, checkLen)
	}
	endTime := time.Now()
	totalTime := endTime.Sub(startTime)
	fmt.Printf("[%d, Att %d] Total execution time for this attempt: %.2fs\n", nOrder, attemptNum, totalTime.Seconds())

	return pairFound
}

func main() {
	// Жёстко заданный путь для выходного файла в контейнере
	outputPath := "/app/output/answer_go_minimal.txt"

	// Создаём директорию если её нет (на случай запуска не в контейнере)
	if err := os.MkdirAll("/app/output", 0755); err != nil {
		log.Printf("Warning: Could not create output directory: %v", err)
	}

	file, err := os.Create(outputPath)
	if err != nil {
		log.Fatalf("FATAL: Could not create output file '%s': %v", outputPath, err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	startOrder := 6
	endOrder := 134
	step := 4

	fmt.Println("Euler Bicircle Search (Go Version with Retry - Minimal File Output)")
	fmt.Printf("Orders: %d to %d, step %d\n", startOrder, endOrder, step)
	fmt.Printf("Parameters: M_BITS=%d, BUCKET_LIMIT=%d, NUM_SEQUENCES_PER_WORKER=%d\n",
		mBits, bucketLimit, numSequencesPerWorker)
	fmt.Printf("Max Retries Per Order: %d\n", maxRetriesPerOrder)
	fmt.Println(strings.Repeat("=", 60))

	for nOrder := startOrder; nOrder <= endOrder; {
		retryCount := 0
		foundPotential := false

		for {
			retryCount++
			fmt.Printf("\n%s\nAttempt %d for order %d\n%s\n",
				strings.Repeat("-", 60), retryCount, nOrder, strings.Repeat("-", 60))

			foundPotential = runForOrder(nOrder, writer, retryCount)

			if foundPotential {
				fmt.Printf("\n%s\nFinished attempts for order %d after finding potential solution on attempt %d.\n%s\n",
					strings.Repeat("=", 60), nOrder, retryCount, strings.Repeat("=", 60))
				nOrder += step
				break
			} else {
				fmt.Printf("\n%s\nOrder %d attempt %d resulted in 'No solution found'.\n",
					strings.Repeat("-", 50), nOrder, retryCount)

				if retryCount >= maxRetriesPerOrder {
					fmt.Printf("Maximum retries (%d) reached for order %d. Moving to next order.\n%s\n",
						maxRetriesPerOrder, nOrder, strings.Repeat("=", 60))
					nOrder += step
					break
				} else {
					fmt.Printf("Retrying order %d...\n%s\n", nOrder, strings.Repeat("-", 50))
				}
			}
		}
	}

	fmt.Printf("\nProcessing complete. Results written to '%s'\n", outputPath)
}
