let max (x: i32) (y: i32): i32 =
  if x > y then x else y

let redOp ((mssx, misx, mcsx, tsx): (i32,i32,i32,i32))
          ((mssy, misy, mcsy, tsy): (i32,i32,i32,i32)): (i32,i32,i32,i32) =
  ( max mssx (max mssy (mcsx + misy))
  , max misx (tsx+misy)
  , max mcsy (mcsx+tsy)
  , tsx + tsy)

let mapOp (x: i32): (i32,i32,i32,i32) =
  ( max x 0
  , max x 0
  , max x 0
  , x)

let main (xs: i32): [](i32) =
 let x = [2883i32, 4893i32, 1063i32, 5838i32, 7800i32, 3587i32, 4281i32, 3601i32, 9561i32, 8886i32]
 let y = [6.595154f32, 3.7526035f32, -2.6016226f32, 6.3009577f32, 5.896557f32, 1.9665699f32, 3.5765095f32, -2.4063258f32, 4.668213f32, 3.1187344f32]
 let w = iota 10
-- in  map(\(i,j,t) -> f32(w[t]) + f32(i)*j) (zip x y w)
 let foo = zip x y
 let (bar, maid) = unzip foo
 in iota 10
