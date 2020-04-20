-- file: net.hs
-- description: a neural net in functional programing
-- contributors: Justin Sostre
-- log: log.txt

import Prelude hiding ((<>))

import Numeric.LinearAlgebra

type ColumnVector a = Matrix a

data PropagatedLayer = PropagatedLayer {
      pIn  :: ColumnVector Float,
      pOut :: ColumnVector Float,
      pFa' :: ColumnVector Float,
      pW   :: Matrix Float,
      pAS  :: ActivationSpec
    }
  | PropagatedSensorLayer {
      pOut :: ColumnVector Float
    }


data ActivationSpec = ActivationSpec {
    asF  :: Float -> Float,
    asF' :: Float -> Float,
    desc :: String
  }


data Layer = Layer {
    lW  :: Matrix Float,
    lAS :: ActivationSpec
  }


data BackpropNet = BackpropNet {
    layers       :: [Layer],
    learningRate :: Float
  }


checkDimensions :: Matrix Float -> Matrix Float -> Matrix Float
checkDimensions w1 w2 =
  if rows w1 == cols w2
     then w2
          else error "Inconsistent dimensions in weight matrix."


buildLayer w s = Layer {lW=w, lAS=s}


buildBackpropNet ::
  Float -> [Matrix Float] -> ActivationSpec -> BackpropNet
buildBackpropNet lr ws s = BackpropNet { layers=ls, learningRate=lr }
                           where checkedWeights = scanl1 checkDimensions ws
                                 ls             = map buildLayer checkedWeights
                                 buildLayer w   = Layer { lW = w, lAS = s}


propagate :: PropagatedLayer -> Layer -> PropagatedLayer
propagate layerJ layerK = PropagatedLayer {
                            pIn  = x,
                            pOut = y,
                            pFa' = fa',
                            pW   = w,
                            pAS  = lAS layerK
                          }
                          where x   = pOut layerJ
                                w   = lW layerK
                                a   = w <> x
                                f   = asF ( lAS layerK )
                                y   = mapMatrix f a
                                f'  = asF' ( lAS layerK )
                                fa' = mapMatrix f' a


-- Checks to make sure input is in [0,1]
checkInput :: ColumnVector Float -> Bool
checkInput m = all (<= 0.5) $ map (subtract 1) $ concat $ toLists m


-- Map matrix using currying
mapMatrix :: (Float -> Float) -> ColumnVector Float -> ColumnVector Float
mapMatrix f m = fromLists $ map (map f) $ toLists m


-- Validate that the input is correct
validateInput :: BackpropNet -> ColumnVector Float -> ColumnVector Float
validateInput net input = if iRows == cRows && checkInput input
                             then input
                                else error "Input is incorrect."
                                     where iRows = rows input
                                           cRows = rows . lW . head .
                                             layers $ net


-- Propagates the network
propagateNet :: ColumnVector Float -> BackpropNet -> [PropagatedLayer]
propagateNet input net = tail calc
                         where calc            = scanl propagate layer0 (layers net)
                               layer0          = PropagatedSensorLayer { pOut=validatedInputs }
                               validatedInputs = validateInput net input


data BackpropagatedLayer = BackpropagatedLayer {
                             bpDazzle  :: ColumnVector Float,
                             bpErrGrad :: ColumnVector Float,
                             bpFa'     :: ColumnVector Float,
                             bpIn      :: ColumnVector Float,
                             bpOut     :: ColumnVector Float,
                             bpW       :: Matrix Float,
                             bpAS      :: ActivationSpec
                           }


backpropagate ::
  PropagatedLayer -> BackpropagatedLayer -> BackpropagatedLayer
backpropagate layerJ layerK = BackpropagatedLayer {
                                bpDazzle  = dazzleJ,
                                bpErrGrad = errorGrad dazzleJ faJ' (pIn layerJ),
                                bpFa'     = pFa' layerJ,
                                bpIn      = pIn layerJ,
                                bpOut     = pOut layerJ,
                                bpW       = pW layerJ,
                                bpAS      = pAS layerJ
                              }
                              where
                                wKT = tr ( bpW layerK )
                                dazzleJ = wKT <> (dazzleK * faK')
                                dazzleK = bpDazzle layerK
                                faK'    = bpFa' layerK
                                faJ'    = pFa' layerJ


errorGrad :: ColumnVector Float -> ColumnVector Float ->
  ColumnVector Float -> Matrix Float
errorGrad dazzle fa' input = (dazzle *fa') <> tr input


backpropagateFinalLayer ::
  PropagatedLayer -> ColumnVector Float -> BackpropagatedLayer
backpropagateFinalLayer l t = BackpropagatedLayer {
                                bpDazzle = dazzle,
                                bpErrGrad = errorGrad dazzle fa' (pIn l),
                                bpFa' = pFa' l,
                                bpIn = pIn l,
                                bpOut = pOut l,
                                bpW = pW l,
                                bpAS = pAS l
                              }
                              where dazzle = pOut l - t
                                    fa'    = pFa' l


backpropagateNet ::
  ColumnVector Float -> [PropagatedLayer] -> [BackpropagatedLayer]
backpropagateNet target layers = scanr backpropagate layerL hiddenLayers
  where hiddenLayers = init layers
        layerL       = backpropagateFinalLayer (last layers) target



update :: Float -> BackpropagatedLayer -> Layer
update rate layer = Layer { lW = wNew, lAS = bpAS layer }
                    where wOld = bpW layer
                          delW = rate `scale` bpErrGrad layer
                          wNew = wOld - delW



