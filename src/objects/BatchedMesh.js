import { BufferAttribute } from '../core/BufferAttribute.js';
import { BufferGeometry } from '../core/BufferGeometry.js';
import { DataTexture } from '../textures/DataTexture.js';
import { FloatType, RedIntegerFormat, UnsignedIntType, RGBAFormat } from '../constants.js';
import { Matrix4, Color, Box3, Sphere, Frustum, Vector3 } from '../math/index.js';
import { Mesh } from './Mesh.js';
import { ColorManagement } from '../math/ColorManagement.js';

// Helper Functions
const ascIdSort = ( a, b ) => a - b;
const sortOpaque = ( a, b ) => a.z - b.z;
const sortTransparent = ( a, b ) => b.z - a.z;

class MultiDrawRenderList {

	constructor() {

		this.index = 0;
		this.pool = [];
		this.list = [];

	}

	push( start, count, z, index ) {

		if ( this.index >= this.pool.length ) {

			this.pool.push( { start: - 1, count: - 1, z: - 1, index: - 1 } );

		}

		const item = this.pool[ this.index ++ ];
		Object.assign( item, { start, count, z, index } );
		this.list.push( item );

	}

	reset() {

		this.list.length = 0;
		this.index = 0;

	}

}

// Reusable Variables
const _matrix = /*@__PURE__*/ new Matrix4();
const _whiteColor = /*@__PURE__*/ new Color( 1, 1, 1 );
const _frustum = /*@__PURE__*/ new Frustum();
const _box = /*@__PURE__*/ new Box3();
const _sphere = /*@__PURE__*/ new Sphere();
const _vector = /*@__PURE__*/ new Vector3();
const _forward = /*@__PURE__*/ new Vector3();
const _temp = /*@__PURE__*/ new Vector3();
const _renderList = /*@__PURE__*/ new MultiDrawRenderList();
const _mesh = /*@__PURE__*/ new Mesh();
const _batchIntersects = [];

// Helper Functions
function copyAttributeData( src, target, targetOffset = 0 ) {

	const itemSize = target.itemSize;

	if ( src.isInterleavedBufferAttribute || src.array.constructor !== target.array.constructor ) {

		const vertexCount = src.count;
		for ( let i = 0; i < vertexCount; i ++ ) {

			for ( let c = 0; c < itemSize; c ++ ) {

				target.setComponent( i + targetOffset, c, src.getComponent( i, c ) );

			}

		}

	} else {

		target.array.set( src.array, targetOffset * itemSize );

	}

	target.needsUpdate = true;

}

function copyArrayContents( src, target ) {

	const len = Math.min( src.length, target.length );

	if ( src.constructor !== target.constructor ) {

		for ( let i = 0; i < len; i ++ ) {

			target[ i ] = src[ i ];

		}

	} else {

		target.set( src.subarray( 0, len ) );

	}

}

class BatchedMesh extends Mesh {

	constructor( maxInstanceCount, maxVertexCount, maxIndexCount = maxVertexCount * 2, material ) {

		super( new BufferGeometry(), material );

		this.isBatchedMesh = true;
		this.perObjectFrustumCulled = true;
		this.sortObjects = true;
		this.boundingBox = null;
		this.boundingSphere = null;
		this.customSort = null;

		// Internal State
		this._instanceInfo = [];
		this._geometryInfo = [];
		this._availableInstanceIds = [];
		this._availableGeometryIds = [];
		this._nextIndexStart = 0;
		this._nextVertexStart = 0;
		this._geometryCount = 0;
		this._visibilityChanged = true;
		this._geometryInitialized = false;

		// User Options
		this._maxInstanceCount = maxInstanceCount;
		this._maxVertexCount = maxVertexCount;
		this._maxIndexCount = maxIndexCount;

		// Buffers for Multi-Draw
		this._multiDrawCounts = new Int32Array( maxInstanceCount );
		this._multiDrawStarts = new Int32Array( maxInstanceCount );
		this._multiDrawCount = 0;

		// Data Textures
		this._initMatricesTexture();
		this._initIndirectTexture();
		this._colorsTexture = null;

	}

	// Getter Methods
	get maxInstanceCount() {

		return this._maxInstanceCount;

	}

	get instanceCount() {

		return this._instanceInfo.length - this._availableInstanceIds.length;

	}

	get unusedVertexCount() {

		return this._maxVertexCount - this._nextVertexStart;

	}

	get unusedIndexCount() {

		return this._maxIndexCount - this._nextIndexStart;

	}

	// Initialization Methods
	_initMatricesTexture() {

		const size = Math.max( Math.ceil( Math.sqrt( this._maxInstanceCount * 4 ) / 4 ) * 4, 4 );
		const matricesArray = new Float32Array( size * size * 4 );
		this._matricesTexture = new DataTexture( matricesArray, size, size, RGBAFormat, FloatType );

	}

	_initIndirectTexture() {

		const size = Math.ceil( Math.sqrt( this._maxInstanceCount ) );
		const indirectArray = new Uint32Array( size * size );
		this._indirectTexture = new DataTexture( indirectArray, size, size, RedIntegerFormat, UnsignedIntType );

	}

	_initColorsTexture() {

		const size = Math.ceil( Math.sqrt( this._maxInstanceCount ) );
		const colorsArray = new Float32Array( size * size * 4 ).fill( 1 );
		this._colorsTexture = new DataTexture( colorsArray, size, size, RGBAFormat, FloatType );
		this._colorsTexture.colorSpace = ColorManagement.workingColorSpace;

	}

	_initializeGeometry( reference ) {

		if ( this._geometryInitialized ) return;

		const geometry = this.geometry;
		const { _maxVertexCount: maxVertexCount, _maxIndexCount: maxIndexCount } = this;

		for ( const attributeName in reference.attributes ) {

			const srcAttribute = reference.getAttribute( attributeName );
			const { array, itemSize, normalized } = srcAttribute;
			const dstArray = new array.constructor( maxVertexCount * itemSize );
			geometry.setAttribute( attributeName, new BufferAttribute( dstArray, itemSize, normalized ) );

		}

		if ( reference.getIndex() ) {

			const indexArray = maxVertexCount > 65535 ? new Uint32Array( maxIndexCount ) : new Uint16Array( maxIndexCount );
			geometry.setIndex( new BufferAttribute( indexArray, 1 ) );

		}

		this._geometryInitialized = true;

	}

	_validateGeometry( geometry ) {

		const batchGeometry = this.geometry;

		if ( !! geometry.getIndex() !== !! batchGeometry.getIndex() ) {

			throw new Error( 'BatchedMesh: All geometries must consistently have "index".' );

		}

		for ( const attributeName in batchGeometry.attributes ) {

			if ( ! geometry.hasAttribute( attributeName ) ) {

				throw new Error( `BatchedMesh: Added geometry missing "${attributeName}". All geometries must have consistent attributes.` );

			}

			const srcAttribute = geometry.getAttribute( attributeName );
			const dstAttribute = batchGeometry.getAttribute( attributeName );

			if ( srcAttribute.itemSize !== dstAttribute.itemSize || srcAttribute.normalized !== dstAttribute.normalized ) {

				throw new Error( 'BatchedMesh: All attributes must have a consistent itemSize and normalized value.' );

			}

		}

	}

	// Validation Methods
	validateInstanceId( instanceId ) {

		const instanceInfo = this._instanceInfo[ instanceId ];
		if ( ! instanceInfo || ! instanceInfo.active ) {

			throw new Error( `BatchedMesh: Invalid instanceId ${instanceId}. Instance is either out of range or has been deleted.` );

		}

	}

	validateGeometryId( geometryId ) {

		const geometryInfo = this._geometryInfo[ geometryId ];
		if ( ! geometryInfo || ! geometryInfo.active ) {

			throw new Error( `BatchedMesh: Invalid geometryId ${geometryId}. Geometry is either out of range or has been deleted.` );

		}

	}

	// Public Methods
	setCustomSort( func ) {

		this.customSort = func;
		return this;

	}

	computeBoundingBox() {

		if ( ! this.boundingBox ) this.boundingBox = new Box3();

		this.boundingBox.makeEmpty();
		for ( const [ i, info ] of this._instanceInfo.entries() ) {

			if ( ! info.active || ! info.visible ) continue;

			const geometryId = info.geometryIndex;
			this.getMatrixAt( i, _matrix );
			this.getBoundingBoxAt( geometryId, _box ).applyMatrix4( _matrix );
			this.boundingBox.union( _box );

		}

	}

	computeBoundingSphere() {

		if ( ! this.boundingSphere ) this.boundingSphere = new Sphere();

		this.boundingSphere.makeEmpty();
		for ( const [ i, info ] of this._instanceInfo.entries() ) {

			if ( ! info.active || ! info.visible ) continue;

			const geometryId = info.geometryIndex;
			this.getMatrixAt( i, _matrix );
			this.getBoundingSphereAt( geometryId, _sphere ).applyMatrix4( _matrix );
			this.boundingSphere.union( _sphere );

		}

	}

	addInstance( geometryId ) {

		if ( this.instanceCount >= this._maxInstanceCount && ! this._availableInstanceIds.length ) {

			throw new Error( 'BatchedMesh: Maximum instance count reached.' );

		}

		const instanceInfo = { visible: true, active: true, geometryIndex: geometryId };
		let instanceId;

		if ( this._availableInstanceIds.length ) {

			this._availableInstanceIds.sort( ascIdSort );
			instanceId = this._availableInstanceIds.shift();
			this._instanceInfo[ instanceId ] = instanceInfo;

		} else {

			instanceId = this._instanceInfo.length;
			this._instanceInfo.push( instanceInfo );

		}

		_matrix.identity().toArray( this._matricesTexture.image.data, instanceId * 16 );
		this._matricesTexture.needsUpdate = true;

		if ( this._colorsTexture ) {

			_whiteColor.toArray( this._colorsTexture.image.data, instanceId * 4 );
			this._colorsTexture.needsUpdate = true;

		}

		this._visibilityChanged = true;
		return instanceId;

	}

	addGeometry( geometry, reservedVertexCount = - 1, reservedIndexCount = - 1 ) {

		this._initializeGeometry( geometry );
		this._validateGeometry( geometry );

		const vertexCount = geometry.getAttribute( 'position' ).count;
		const index = geometry.getIndex();
		const indexCount = index ? index.count : 0;
		const geometryInfo = {
			vertexStart: this._nextVertexStart,
			vertexCount: 0,
			reservedVertexCount: reservedVertexCount === - 1 ? vertexCount : reservedVertexCount,
			indexStart: this._nextIndexStart,
			indexCount: 0,
			reservedIndexCount: reservedIndexCount === - 1 ? indexCount : reservedIndexCount,
			start: - 1,
			count: - 1,
			boundingBox: null,
			boundingSphere: null,
			active: true,
		};

		if (
			geometryInfo.indexStart + geometryInfo.reservedIndexCount > this._maxIndexCount ||
      geometryInfo.vertexStart + geometryInfo.reservedVertexCount > this._maxVertexCount
		) {

			throw new Error( 'BatchedMesh: Reserved space request exceeds the maximum buffer size.' );

		}

		let geometryId;

		if ( this._availableGeometryIds.length ) {

			this._availableGeometryIds.sort( ascIdSort );
			geometryId = this._availableGeometryIds.shift();
			this._geometryInfo[ geometryId ] = geometryInfo;

		} else {

			geometryId = this._geometryCount ++;
			this._geometryInfo.push( geometryInfo );

		}

		this.setGeometryAt( geometryId, geometry );

		this._nextIndexStart = geometryInfo.indexStart + geometryInfo.reservedIndexCount;
		this._nextVertexStart = geometryInfo.vertexStart + geometryInfo.reservedVertexCount;

		return geometryId;

	}

	setGeometryAt( geometryId, geometry ) {

		this.validateGeometryId( geometryId );
		this._validateGeometry( geometry );

		const batchGeometry = this.geometry;
		const geometryInfo = this._geometryInfo[ geometryId ];
		const { vertexStart, reservedVertexCount } = geometryInfo;
		geometryInfo.vertexCount = geometry.getAttribute( 'position' ).count;

		for ( const attributeName in batchGeometry.attributes ) {

			const srcAttribute = geometry.getAttribute( attributeName );
			const dstAttribute = batchGeometry.getAttribute( attributeName );
			copyAttributeData( srcAttribute, dstAttribute, vertexStart );

			const itemSize = srcAttribute.itemSize;
			for ( let i = srcAttribute.count; i < reservedVertexCount; i ++ ) {

				const index = vertexStart + i;
				for ( let c = 0; c < itemSize; c ++ ) {

					dstAttribute.setComponent( index, c, 0 );

				}

			}

			dstAttribute.needsUpdate = true;
			dstAttribute.updateRange.set( vertexStart * itemSize, reservedVertexCount * itemSize );

		}

		if ( batchGeometry.index ) {

			const indexStart = geometryInfo.indexStart;
			const reservedIndexCount = geometryInfo.reservedIndexCount;
			const srcIndex = geometry.getIndex();
			const dstIndex = batchGeometry.getIndex();
			geometryInfo.indexCount = srcIndex.count;

			for ( let i = 0; i < srcIndex.count; i ++ ) {

				dstIndex.setX( indexStart + i, vertexStart + srcIndex.getX( i ) );

			}

			for ( let i = srcIndex.count; i < reservedIndexCount; i ++ ) {

				dstIndex.setX( indexStart + i, vertexStart );

			}

			dstIndex.needsUpdate = true;
			dstIndex.updateRange.set( indexStart, reservedIndexCount );

		}

		geometryInfo.start = batchGeometry.index ? geometryInfo.indexStart : geometryInfo.vertexStart;
		geometryInfo.count = batchGeometry.index ? geometryInfo.indexCount : geometryInfo.vertexCount;

		geometryInfo.boundingBox = geometry.boundingBox ? geometry.boundingBox.clone() : null;
		geometryInfo.boundingSphere = geometry.boundingSphere ? geometry.boundingSphere.clone() : null;

		this._visibilityChanged = true;
		return geometryId;

	}

	deleteGeometry( geometryId ) {

		this.validateGeometryId( geometryId );

		this._instanceInfo.forEach( ( info, index ) => {

			if ( info.geometryIndex === geometryId ) {

				this.deleteInstance( index );

			}

		} );

		this._geometryInfo[ geometryId ].active = false;
		this._availableGeometryIds.push( geometryId );
		this._visibilityChanged = true;

		return this;

	}

	deleteInstance( instanceId ) {

		this.validateInstanceId( instanceId );

		this._instanceInfo[ instanceId ].active = false;
		this._availableInstanceIds.push( instanceId );
		this._visibilityChanged = true;

		return this;

	}

	optimize() {

		let nextVertexStart = 0;
		let nextIndexStart = 0;

		const geometryInfoList = this._geometryInfo;
		const indices = geometryInfoList
			.map( ( _, i ) => i )
			.filter( ( i ) => geometryInfoList[ i ].active )
			.sort( ( a, b ) => geometryInfoList[ a ].vertexStart - geometryInfoList[ b ].vertexStart );

		const geometry = this.geometry;
		for ( const index of indices ) {

			const geometryInfo = geometryInfoList[ index ];

			if ( geometry.index && geometryInfo.indexStart !== nextIndexStart ) {

				const { indexStart, vertexStart, reservedIndexCount } = geometryInfo;
				const indexArray = geometry.index.array;

				const elementDelta = nextVertexStart - vertexStart;
				for ( let j = indexStart; j < indexStart + reservedIndexCount; j ++ ) {

					indexArray[ j ] += elementDelta;

				}

				indexArray.copyWithin( nextIndexStart, indexStart, indexStart + reservedIndexCount );
				geometry.index.updateRange.set( nextIndexStart, reservedIndexCount );

				geometryInfo.indexStart = nextIndexStart;

			}

			if ( geometryInfo.vertexStart !== nextVertexStart ) {

				const { vertexStart, reservedVertexCount } = geometryInfo;
				for ( const key in geometry.attributes ) {

					const attribute = geometry.attributes[ key ];
					const { array, itemSize } = attribute;
					array.copyWithin(
						nextVertexStart * itemSize,
						vertexStart * itemSize,
						( vertexStart + reservedVertexCount ) * itemSize
					);
					attribute.updateRange.set( nextVertexStart * itemSize, reservedVertexCount * itemSize );

				}

				geometryInfo.vertexStart = nextVertexStart;

			}

			nextIndexStart += geometryInfo.reservedIndexCount;
			nextVertexStart += geometryInfo.reservedVertexCount;
			geometryInfo.start = geometry.index ? geometryInfo.indexStart : geometryInfo.vertexStart;

		}

		this._nextIndexStart = nextIndexStart;
		this._nextVertexStart = nextVertexStart;

		return this;

	}

	getBoundingBoxAt( geometryId, target ) {

		this.validateGeometryId( geometryId );
		const geometryInfo = this._geometryInfo[ geometryId ];

		if ( ! geometryInfo.boundingBox ) {

			const geometry = this.geometry;
			const position = geometry.attributes.position;
			const index = geometry.index;
			const start = geometryInfo.start;
			const count = geometryInfo.count;

			const box = new Box3();
			for ( let i = start; i < start + count; i ++ ) {

				const idx = index ? index.getX( i ) : i;
				box.expandByPoint( _vector.fromBufferAttribute( position, idx ) );

			}

			geometryInfo.boundingBox = box;

		}

		target.copy( geometryInfo.boundingBox );
		return target;

	}

	getBoundingSphereAt( geometryId, target ) {

		this.validateGeometryId( geometryId );
		const geometryInfo = this._geometryInfo[ geometryId ];

		if ( ! geometryInfo.boundingSphere ) {

			this.getBoundingBoxAt( geometryId, _box );
			_box.getCenter( _sphere.center );

			const geometry = this.geometry;
			const position = geometry.attributes.position;
			const index = geometry.index;
			const start = geometryInfo.start;
			const count = geometryInfo.count;

			let maxRadiusSq = 0;
			for ( let i = start; i < start + count; i ++ ) {

				const idx = index ? index.getX( i ) : i;
				_vector.fromBufferAttribute( position, idx );
				maxRadiusSq = Math.max( maxRadiusSq, _sphere.center.distanceToSquared( _vector ) );

			}

			_sphere.radius = Math.sqrt( maxRadiusSq );
			geometryInfo.boundingSphere = _sphere.clone();

		}

		target.copy( geometryInfo.boundingSphere );
		return target;

	}

	setMatrixAt( instanceId, matrix ) {

		this.validateInstanceId( instanceId );
		matrix.toArray( this._matricesTexture.image.data, instanceId * 16 );
		this._matricesTexture.needsUpdate = true;
		return this;

	}

	getMatrixAt( instanceId, matrix ) {

		this.validateInstanceId( instanceId );
		return matrix.fromArray( this._matricesTexture.image.data, instanceId * 16 );

	}

	setColorAt( instanceId, color ) {

		this.validateInstanceId( instanceId );
		if ( ! this._colorsTexture ) this._initColorsTexture();

		color.toArray( this._colorsTexture.image.data, instanceId * 4 );
		this._colorsTexture.needsUpdate = true;
		return this;

	}

	getColorAt( instanceId, color ) {

		this.validateInstanceId( instanceId );
		return color.fromArray( this._colorsTexture.image.data, instanceId * 4 );

	}

	setVisibleAt( instanceId, value ) {

		this.validateInstanceId( instanceId );
		if ( this._instanceInfo[ instanceId ].visible === value ) return this;

		this._instanceInfo[ instanceId ].visible = value;
		this._visibilityChanged = true;
		return this;

	}

	getVisibleAt( instanceId ) {

		this.validateInstanceId( instanceId );
		return this._instanceInfo[ instanceId ].visible;

	}

	setGeometryIdAt( instanceId, geometryId ) {

		this.validateInstanceId( instanceId );
		this.validateGeometryId( geometryId );

		this._instanceInfo[ instanceId ].geometryIndex = geometryId;
		return this;

	}

	getGeometryIdAt( instanceId ) {

		this.validateInstanceId( instanceId );
		return this._instanceInfo[ instanceId ].geometryIndex;

	}

	getGeometryRangeAt( geometryId, target = {} ) {

		this.validateGeometryId( geometryId );
		const geometryInfo = this._geometryInfo[ geometryId ];

		Object.assign( target, {
			vertexStart: geometryInfo.vertexStart,
			vertexCount: geometryInfo.vertexCount,
			reservedVertexCount: geometryInfo.reservedVertexCount,
			indexStart: geometryInfo.indexStart,
			indexCount: geometryInfo.indexCount,
			reservedIndexCount: geometryInfo.reservedIndexCount,
			start: geometryInfo.start,
			count: geometryInfo.count,
		} );

		return target;

	}

	setInstanceCount( maxInstanceCount ) {

		const availableInstanceIds = this._availableInstanceIds;
		const instanceInfo = this._instanceInfo;
		availableInstanceIds.sort( ascIdSort );
		while ( availableInstanceIds[ availableInstanceIds.length - 1 ] === instanceInfo.length - 1 ) {

			instanceInfo.pop();
			availableInstanceIds.pop();

		}

		if ( maxInstanceCount < instanceInfo.length ) {

			throw new Error( `BatchedMesh: Instance ids outside the range ${maxInstanceCount} are being used. Cannot shrink instance count.` );

		}

		const multiDrawCounts = new Int32Array( maxInstanceCount );
		const multiDrawStarts = new Int32Array( maxInstanceCount );
		copyArrayContents( this._multiDrawCounts, multiDrawCounts );
		copyArrayContents( this._multiDrawStarts, multiDrawStarts );

		this._multiDrawCounts = multiDrawCounts;
		this._multiDrawStarts = multiDrawStarts;
		this._maxInstanceCount = maxInstanceCount;

		const indirectTexture = this._indirectTexture;
		const matricesTexture = this._matricesTexture;
		const colorsTexture = this._colorsTexture;

		indirectTexture.dispose();
		this._initIndirectTexture();
		copyArrayContents( indirectTexture.image.data, this._indirectTexture.image.data );

		matricesTexture.dispose();
		this._initMatricesTexture();
		copyArrayContents( matricesTexture.image.data, this._matricesTexture.image.data );

		if ( colorsTexture ) {

			colorsTexture.dispose();
			this._initColorsTexture();
			copyArrayContents( colorsTexture.image.data, this._colorsTexture.image.data );

		}

	}

	setGeometrySize( maxVertexCount, maxIndexCount ) {

		const validRanges = this._geometryInfo.filter( ( info ) => info.active );
		const requiredVertexLength = Math.max( ...validRanges.map( ( range ) => range.vertexStart + range.reservedVertexCount ) );

		if ( requiredVertexLength > maxVertexCount ) {

			throw new Error( `BatchedMesh: Geometry vertex values are being used outside the range ${maxVertexCount}. Cannot shrink further.` );

		}

		if ( this.geometry.index ) {

			const requiredIndexLength = Math.max( ...validRanges.map( ( range ) => range.indexStart + range.reservedIndexCount ) );
			if ( requiredIndexLength > maxIndexCount ) {

				throw new Error( `BatchedMesh: Geometry index values are being used outside the range ${maxIndexCount}. Cannot shrink further.` );

			}

		}

		const oldGeometry = this.geometry;
		oldGeometry.dispose();

		this._maxVertexCount = maxVertexCount;
		this._maxIndexCount = maxIndexCount;

		if ( this._geometryInitialized ) {

			this._geometryInitialized = false;
			this.geometry = new BufferGeometry();
			this._initializeGeometry( oldGeometry );

		}

		const geometry = this.geometry;
		if ( oldGeometry.index ) {

			copyArrayContents( oldGeometry.index.array, geometry.index.array );

		}

		for ( const key in oldGeometry.attributes ) {

			copyArrayContents( oldGeometry.attributes[ key ].array, geometry.attributes[ key ].array );

		}

	}

	raycast( raycaster, intersects ) {

		const instanceInfo = this._instanceInfo;
		const geometryInfoList = this._geometryInfo;
		const matrixWorld = this.matrixWorld;
		const batchGeometry = this.geometry;

		_mesh.material = this.material;
		_mesh.geometry.index = batchGeometry.index;
		_mesh.geometry.attributes = batchGeometry.attributes;
		if ( _mesh.geometry.boundingBox === null ) _mesh.geometry.boundingBox = new Box3();
		if ( _mesh.geometry.boundingSphere === null ) _mesh.geometry.boundingSphere = new Sphere();

		for ( let i = 0, l = instanceInfo.length; i < l; i ++ ) {

			const info = instanceInfo[ i ];
			if ( ! info.visible || ! info.active ) continue;

			const geometryId = info.geometryIndex;
			const geometryInfo = geometryInfoList[ geometryId ];
			_mesh.geometry.setDrawRange( geometryInfo.start, geometryInfo.count );

			this.getMatrixAt( i, _mesh.matrixWorld ).premultiply( matrixWorld );
			this.getBoundingBoxAt( geometryId, _mesh.geometry.boundingBox );
			this.getBoundingSphereAt( geometryId, _mesh.geometry.boundingSphere );
			_mesh.raycast( raycaster, _batchIntersects );

			for ( const intersect of _batchIntersects ) {

				intersect.object = this;
				intersect.batchId = i;
				intersects.push( intersect );

			}

			_batchIntersects.length = 0;

		}

		_mesh.material = null;
		_mesh.geometry.index = null;
		_mesh.geometry.attributes = {};
		_mesh.geometry.setDrawRange( 0, Infinity );

	}

	copy( source ) {

		super.copy( source );

		this.geometry = source.geometry.clone();
		this.perObjectFrustumCulled = source.perObjectFrustumCulled;
		this.sortObjects = source.sortObjects;
		this.boundingBox = source.boundingBox ? source.boundingBox.clone() : null;
		this.boundingSphere = source.boundingSphere ? source.boundingSphere.clone() : null;

		this._geometryInfo = source._geometryInfo.map( ( info ) => ( {
			...info,
			boundingBox: info.boundingBox ? info.boundingBox.clone() : null,
			boundingSphere: info.boundingSphere ? info.boundingSphere.clone() : null,
		} ) );
		this._instanceInfo = source._instanceInfo.map( ( info ) => ( { ...info } ) );

		this._maxInstanceCount = source._maxInstanceCount;
		this._maxVertexCount = source._maxVertexCount;
		this._maxIndexCount = source._maxIndexCount;

		this._geometryInitialized = source._geometryInitialized;
		this._geometryCount = source._geometryCount;
		this._multiDrawCounts = source._multiDrawCounts.slice();
		this._multiDrawStarts = source._multiDrawStarts.slice();

		this._matricesTexture = source._matricesTexture.clone();
		this._matricesTexture.image.data = this._matricesTexture.image.data.slice();

		if ( source._colorsTexture ) {

			this._colorsTexture = source._colorsTexture.clone();
			this._colorsTexture.image.data = this._colorsTexture.image.data.slice();

		}

		return this;

	}

	dispose() {

		this.geometry.dispose();

		this._matricesTexture.dispose();
		this._matricesTexture = null;

		this._indirectTexture.dispose();
		this._indirectTexture = null;

		if ( this._colorsTexture ) {

			this._colorsTexture.dispose();
			this._colorsTexture = null;

		}

		return this;

	}

	onBeforeRender( renderer, scene, camera, geometry, material ) {

		if ( ! this._visibilityChanged && ! this.perObjectFrustumCulled && ! this.sortObjects ) return;

		const index = geometry.getIndex();
		const bytesPerElement = index === null ? 1 : index.array.BYTES_PER_ELEMENT;

		const instanceInfo = this._instanceInfo;
		const multiDrawStarts = this._multiDrawStarts;
		const multiDrawCounts = this._multiDrawCounts;
		const geometryInfoList = this._geometryInfo;
		const perObjectFrustumCulled = this.perObjectFrustumCulled;
		const indirectTexture = this._indirectTexture;
		const indirectArray = indirectTexture.image.data;

		if ( perObjectFrustumCulled ) {

			_matrix.multiplyMatrices( camera.projectionMatrix, camera.matrixWorldInverse ).multiply( this.matrixWorld );
			_frustum.setFromProjectionMatrix( _matrix, renderer.coordinateSystem );

		}

		let multiDrawCount = 0;
		if ( this.sortObjects ) {

			_matrix.copy( this.matrixWorld ).invert();
			_vector.setFromMatrixPosition( camera.matrixWorld ).applyMatrix4( _matrix );
			_forward.set( 0, 0, - 1 ).transformDirection( camera.matrixWorld ).transformDirection( _matrix );

			for ( let i = 0, l = instanceInfo.length; i < l; i ++ ) {

				const info = instanceInfo[ i ];
				if ( ! info.visible || ! info.active ) continue;

				const geometryId = info.geometryIndex;
				this.getMatrixAt( i, _matrix );
				this.getBoundingSphereAt( geometryId, _sphere ).applyMatrix4( _matrix );

				let culled = false;
				if ( perObjectFrustumCulled ) {

					culled = ! _frustum.intersectsSphere( _sphere );

				}

				if ( ! culled ) {

					const geometryInfo = geometryInfoList[ geometryId ];
					const z = _temp.subVectors( _sphere.center, _vector ).dot( _forward );
					_renderList.push( geometryInfo.start, geometryInfo.count, z, i );

				}

			}

			const list = _renderList.list;
			const customSort = this.customSort;
			if ( customSort === null ) {

				list.sort( material.transparent ? sortTransparent : sortOpaque );

			} else {

				customSort.call( this, list, camera );

			}

			for ( const item of list ) {

				multiDrawStarts[ multiDrawCount ] = item.start * bytesPerElement;
				multiDrawCounts[ multiDrawCount ] = item.count;
				indirectArray[ multiDrawCount ] = item.index;
				multiDrawCount ++;

			}

			_renderList.reset();

		} else {

			for ( let i = 0, l = instanceInfo.length; i < l; i ++ ) {

				const info = instanceInfo[ i ];
				if ( ! info.visible || ! info.active ) continue;

				const geometryId = info.geometryIndex;
				let culled = false;
				if ( perObjectFrustumCulled ) {

					this.getMatrixAt( i, _matrix );
					this.getBoundingSphereAt( geometryId, _sphere ).applyMatrix4( _matrix );
					culled = ! _frustum.intersectsSphere( _sphere );

				}

				if ( ! culled ) {

					const geometryInfo = geometryInfoList[ geometryId ];
					multiDrawStarts[ multiDrawCount ] = geometryInfo.start * bytesPerElement;
					multiDrawCounts[ multiDrawCount ] = geometryInfo.count;
					indirectArray[ multiDrawCount ] = i;
					multiDrawCount ++;

				}

			}

		}

		indirectTexture.needsUpdate = true;
		this._multiDrawCount = multiDrawCount;
		this._visibilityChanged = false;

	}

	onBeforeShadow( renderer, object, camera, shadowCamera, geometry, depthMaterial ) {

		this.onBeforeRender( renderer, null, shadowCamera, geometry, depthMaterial );

	}

}

export { BatchedMesh };
