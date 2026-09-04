-- Augmented Vector Space Playground v2
-- No dependencies — runs on play.haskell.org (base only)

module Main where

import Data.List (sortBy, intercalate)
import Data.Ord  (comparing)

-- ─── Types ───────────────────────────────────────────────────────────────────

type Sample = Double
type Vec    = [Double]
type Label  = Int

data AugConfig = AugConfig
  { lagWindow  :: Int
  , derivOrder :: Int
  , rollingWin :: Int
  } deriving (Show)

data AugPoint = AugPoint
  { apLabel  :: Label
  , apVec    :: Vec
  , apSignal :: String
  } deriving (Show)

data Metric = L2 | Cosine deriving (Show, Eq)

-- ─── Signal generators ───────────────────────────────────────────────────────

sineWave :: Int -> [Sample]
sineWave n = [ sin (2 * pi * fromIntegral t / 20) * 3 | t <- [0..n-1] ]

stepWave :: Int -> [Sample]
stepWave n = [ if t < 20 then 1 else if t < 40 then -1 else 2 | t <- [0..n-1] ]

noisyWave :: Int -> [Sample]
noisyWave n = zipWith (+) (sineWave n)
  [ 0.5 * sin (fromIntegral t * 1.7 + 0.3) | t <- [0..n-1] ]

-- ─── Feature extractors ──────────────────────────────────────────────────────

lagFeats :: Int -> [Sample] -> [(Int, Vec)]
lagFeats p xs =
  [ (i, reverse (take p (drop (i - p + 1) xs)))
  | i <- [p-1 .. length xs - 1] ]

diffs :: Int -> [Sample] -> [Sample]
diffs 0 xs = xs
diffs d xs = diffs (d-1) (zipWith (-) (drop 1 xs) xs)

rollingStats :: Int -> [Sample] -> [(Double, Double)]
rollingStats w xs =
  [ let win = take w (drop i xs)
        mu  = sum win / fromIntegral w
        var = sum (map (\x -> (x - mu)^2) win) / fromIntegral w
    in (mu, sqrt var)
  | i <- [0 .. length xs - w] ]

-- ─── Augmentation ────────────────────────────────────────────────────────────

augment :: AugConfig -> String -> [Sample] -> [AugPoint]
augment cfg name xs = zipWith3 build lagPairs diffSeries statSeries
  where
    p = lagWindow  cfg
    d = derivOrder cfg
    w = rollingWin cfg

    lagPairs   = lagFeats p xs
    start      = p - 1

    diffXs     = diffs 1 xs
    diff2Xs    = diffs 2 xs
    diffSeries = case d of
      0 -> repeat []
      1 -> map (:[]) (drop start diffXs)
      _ -> zipWith (\a b -> [a,b]) (drop start diffXs) (drop start diff2Xs)

    rawStats   = rollingStats w xs
    statSeries = drop (start - (w-1)) rawStats

    build (idx, lags) ds (mu, sigma) =
      AugPoint idx (lags ++ ds ++ [mu, sigma]) name

-- ─── Distance functions ───────────────────────────────────────────────────────

l2 :: Vec -> Vec -> Double
l2 a b = sqrt . sum $ zipWith (\x y -> (x-y)^2) a b

cosine :: Vec -> Vec -> Double
cosine a b =
  let dot = sum (zipWith (*) a b)
      na  = sqrt (sum (map (^2) a))
      nb  = sqrt (sum (map (^2) b))
      den = na * nb
  in  if den < 1e-12 then 1.0 else 1.0 - dot / den

applyMetric :: Metric -> Vec -> Vec -> Double
applyMetric L2     = l2
applyMetric Cosine = cosine

-- ─── kNN ─────────────────────────────────────────────────────────────────────

knn :: Metric -> Int -> AugPoint -> [AugPoint] -> [(AugPoint, Double)]
knn metric k query corpus =
  take k
  . sortBy (comparing snd)
  . map (\p -> (p, applyMetric metric (apVec query) (apVec p)))
  . filter ((/= apLabel query) . apLabel)
  $ corpus

-- ─── Linear regression (OLS) ─────────────────────────────────────────────────

dotV :: Vec -> Vec -> Double
dotV a b = sum (zipWith (*) a b)

cdStep :: Vec -> [(Vec, Double)] -> Vec
cdStep beta pairs =
  foldl updateCoord beta [0..length beta - 1]
  where
    updateCoord b i =
      let xi  = map (\(x,_) -> x !! i) pairs
          ri  = map (\(x,y) -> y - dotV b x + (b !! i) * (x !! i)) pairs
          num = dotV xi ri
          den = dotV xi xi
      in  if abs den < 1e-12 then b
          else take i b ++ [num / den] ++ drop (i+1) b

fitOLS :: Int -> [(Vec, Double)] -> Vec
fitOLS iters pairs =
  case pairs of
    []           -> []
    ((v,_) : _) -> iterate (`cdStep` pairs) (replicate (length v) 0.0) !! iters

predictOLS :: Vec -> Vec -> Double
predictOLS = dotV

regressionPairs :: [Sample] -> [AugPoint] -> [(Vec, Double)]
regressionPairs xs pts =
  [ (apVec p, xs !! (apLabel p + 1))
  | p <- pts
  , apLabel p + 1 < length xs ]

-- ─── Distance matrix ─────────────────────────────────────────────────────────

distMatrix :: Metric -> [AugPoint] -> [[Double]]
distMatrix metric pts =
  [ [ applyMetric metric (apVec a) (apVec b) | b <- pts ] | a <- pts ]

-- ─── Pretty printing ─────────────────────────────────────────────────────────

rule :: String
rule = "└" ++ replicate 66 '─' ++ "┘"

printSection :: String -> [String] -> IO ()
printSection hdr rows = do
  putStrLn ""
  putStrLn $ "┌─ " ++ hdr ++ " " ++ replicate (64 - length hdr) '─' ++ "┐"
  mapM_ (putStrLn . ("│  " ++)) rows
  putStrLn rule

bar :: Double -> Double -> Int -> String
bar v maxV w =
  let len = max 0 . min w $ round (v / max maxV 1e-9 * fromIntegral w)
  in replicate len '█' ++ replicate (w - len) '░'

fmtF :: Int -> Double -> String
fmtF dp x = show (fromIntegral (round (x * 10^dp)) / 10^dp :: Double)

padL :: Int -> String -> String
padL n s = replicate (n - length s) ' ' ++ s

padR :: Int -> String -> String
padR n s = take n (s ++ repeat ' ')

showVec :: Int -> Vec -> String
showVec n vs = "[" ++ intercalate ", " (map (padL 6 . fmtF 2) (take n vs))
            ++ (if length vs > n then ", …]" else "]")

-- ─── Report: signals side by side ────────────────────────────────────────────

reportSignals :: AugConfig -> IO ()
reportSignals cfg = do
  let signals = [("sine", sineWave 60), ("step", stepWave 60), ("noisy", noisyWave 60)]
      w       = 24
  printSection "Signals in augmented vector space (norm per point)" $
    concatMap (\(name, xs) ->
      let pts  = augment cfg name xs
          ns   = map (\p -> (apLabel p, l2 (apVec p) (replicate (length (apVec p)) 0))) pts
          maxN = maximum (map snd ns)
          rows = map (\(t,n) ->
                   "  " ++ padR 5 (show t)
                   ++ bar n maxN w
                   ++ "  " ++ fmtF 2 n) (take 12 ns)
      in ("" : ("── " ++ name ++ " ──") : rows)
    ) signals

-- ─── Report: kNN with both metrics ───────────────────────────────────────────

reportKNN :: [AugPoint] -> IO ()
reportKNN pts = do
  let queryT = 10
  case filter ((== queryT) . apLabel) pts of
    []        -> putStrLn "Query point not found."
    (query:_) -> do
      let runKNN metric = do
            let nbs  = knn metric 5 query pts
                maxD = maximum (map snd nbs)
            printSection ("5-NN of t=" ++ show queryT ++ " — " ++ show metric) $
              [ "Query vec : " ++ showVec 4 (apVec query) ++ " …"
              , "Signal    : " ++ apSignal query
              , "" ] ++
              map (\(p,d) ->
                "t=" ++ padR 3 (show (apLabel p))
                ++ " [" ++ padR 5 (apSignal p) ++ "]  "
                ++ bar d maxD 28
                ++ "  " ++ fmtF 4 d
                ) nbs
      runKNN L2
      runKNN Cosine

-- ─── Report: linear regression ───────────────────────────────────────────────

reportRegression :: [Sample] -> [AugPoint] -> IO ()
reportRegression xs pts = do
  let pairs  = regressionPairs xs pts
      beta   = fitOLS 200 pairs
      preds  = map (\(v,_) -> predictOLS beta v) pairs
      actual = map snd pairs
      errs   = zipWith (\p a -> p - a) preds actual
      mse    = sum (map (^2) errs) / fromIntegral (length errs)
      rmse   = sqrt mse
      maxE   = maximum (map abs errs)
  printSection "OLS regression on augmented vectors (next-step prediction)" $
    [ "Coefficients β : " ++ showVec 6 beta
    , ""
    , "RMSE           : " ++ fmtF 4 rmse
    , "Max |error|    : " ++ fmtF 4 maxE
    , ""
    , "  t      actual  predicted      error" ] ++
    zipWith3 (\p a e ->
      "  " ++ padR 4 (show (apLabel p))
      ++ padL 10 (fmtF 3 a)
      ++ padL 11 (fmtF 3 (predictOLS beta (apVec p)))
      ++ padL 11 (fmtF 4 e)
      ) (take 10 pts) (take 10 actual) (take 10 errs)

-- ─── Report: distance matrix ─────────────────────────────────────────────────

reportDistMatrix :: Metric -> [AugPoint] -> IO ()
reportDistMatrix metric pts = do
  let queries = filter (\p -> apLabel p `elem` [5,10,15,20,25,30]) pts
      mat     = distMatrix metric queries
      header  = "      " ++ concatMap (\p -> padL 7 ("t=" ++ show (apLabel p))) queries
      rows    = zipWith (\p row ->
                  padR 6 ("t=" ++ show (apLabel p))
                  ++ concatMap (\d -> padL 7 (fmtF 3 d)) row
                ) queries mat
  printSection ("Distance matrix — " ++ show metric
             ++ " (query points t=5,10,15,20,25,30)")
    (header : "" : rows)

-- ─── Main ────────────────────────────────────────────────────────────────────

main :: IO ()
main = do
  putStrLn "╔════════════════════════════════════════════════════════════════════╗"
  putStrLn "║     Augmented Vector Space Playground v2  ·  Haskell (base)      ║"
  putStrLn "╚════════════════════════════════════════════════════════════════════╝"

  let cfg    = AugConfig { lagWindow = 3, derivOrder = 1, rollingWin = 4 }
      xs     = sineWave 60
      allPts = concatMap (\(name, sig) -> augment cfg name sig)
                 [ ("sine",  sineWave  60)
                 , ("step",  stepWave  60)
                 , ("noisy", noisyWave 60) ]
      sinePts = augment cfg "sine" xs

  reportSignals    cfg
  reportKNN        allPts
  reportRegression xs sinePts
  reportDistMatrix L2     sinePts
  reportDistMatrix Cosine sinePts

  putStrLn ""
  putStrLn "── Suggested experiments ───────────────────────────────────────────"
  putStrLn "   · Swap sineWave for stepWave in regression"
  putStrLn "   · Increase lagWindow to 6 and observe the variance report"
  putStrLn "   · Change queryT in reportKNN to explore different neighbourhoods"